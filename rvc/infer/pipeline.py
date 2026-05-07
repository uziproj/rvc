import os
import sys
import torch
import faiss

import numpy as np
import torch.nn.functional as F

from scipy import signal

# Import RVC modules - these will work when the package is installed via pip
from rvc.lib.predictor.generator import Generator
from rvc.lib.backend.rms import RMSEnergyExtractor
from rvc.utils import change_rms, clear_gpu_cache
from rvc.lib.config import Config

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

class Pipeline:
    def __init__(self, tgt_sr, config):
        self.x_pad, self.x_query, self.x_center, self.x_max = config.device_config()
        self.sample_rate = 16000
        self.window = 160
        self.t_pad = self.sample_rate * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sample_rate * self.x_query
        self.t_center = self.sample_rate * self.x_center
        self.t_max = self.sample_rate * self.x_max
        self.time_step = self.window / self.sample_rate * 1000
        self.f0_min = 50
        self.f0_max = 1100
        self.device = config.device
        self.is_half = config.is_half
        self.config = config

    def voice_conversion(self, model, net_g, sid, audio0, pitch, pitchf, index, big_npy, index_rate, version, protect, energy):
        # Check if audio0 is too short
        if len(audio0) < 160:  # Minimum length for processing
            return np.zeros(max(0, len(audio0) - 2 * self.t_pad_tgt), dtype=np.float32)
        
        feats = (torch.from_numpy(audio0).half() if self.is_half else torch.from_numpy(audio0).float())
        pitch_guidance = pitch is not None and pitchf is not None
        energy_use = energy is not None

        if feats.dim() == 2: 
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)

        with torch.no_grad():
            padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)
            
            # Add check for very short audio
            if feats.shape[1] < 10:  # Model requires minimum input length
                return np.zeros(max(0, len(audio0) - 2 * self.t_pad_tgt), dtype=np.float32)
                
            logits = model.extract_features(**{"source": feats.to(self.device), "padding_mask": padding_mask, "output_layer": 9 if version == "v1" else 12})
            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]

            if protect < 0.5 and pitch_guidance: 
                feats0 = feats.clone()

            if (index is not None and big_npy is not None and index_rate != 0):
                npy = feats[0].cpu().numpy()
                if self.is_half: 
                    npy = npy.astype(np.float32)

                score, ix = index.search(npy, k=8)
                weight = np.square(1 / score)

                npy = np.sum(big_npy[ix] * np.expand_dims(weight / weight.sum(axis=1, keepdims=True), axis=2), axis=1)
                if self.is_half: 
                    npy = npy.astype(np.float16)

                feats = (torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate + (1 - index_rate) * feats)

            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
            if protect < 0.5 and pitch_guidance: 
                feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
            
            p_len = audio0.shape[0] // self.window

            if feats.shape[1] < p_len:
                p_len = feats.shape[1]
                if pitch_guidance: 
                    pitch, pitchf = pitch[:, :p_len], pitchf[:, :p_len]
                if energy_use: 
                    energy = energy[:, :p_len]

            if protect < 0.5 and pitch_guidance:
                pitchff = pitchf.clone()
                pitchff[pitchf > 0] = 1
                pitchff[pitchf < 1] = protect
                pitchff = pitchff.unsqueeze(-1)

                feats = (feats * pitchff + feats0 * (1 - pitchff)).to(feats0.dtype)

            p_len = torch.tensor([p_len], device=self.device).long()
            feats = feats.half() if self.is_half else feats.float()

            if not pitch_guidance: 
                pitch, pitchf = None, None
            else: 
                pitchf = pitchf.half() if self.is_half else pitchf.float()
            if not energy_use: 
                energy = None
            else: 
                energy = energy.half() if self.is_half else energy.float()

            audio1 = (
                (
                    net_g.infer(
                        feats, 
                        p_len, 
                        pitch, 
                        pitchf,
                        sid,
                        energy
                    )[0][0, 0]
                ).data.cpu().float().numpy()
            )

        del feats, p_len, padding_mask
        clear_gpu_cache()
        return audio1
    
    def pipeline(
        self, 
        model, 
        net_g, 
        sid, 
        audio, 
        f0_up_key, 
        f0_method, 
        file_index, 
        index_rate, 
        pitch_guidance, 
        filter_radius, 
        volume_envelope, 
        version, 
        protect, 
        hop_length, 
        energy_use=False,
        f0_autotune=False, 
        f0_autotune_strength=1.0,
        proposal_pitch=False,
        proposal_pitch_threshold=255.0
    ):
        
        if file_index != "" and os.path.exists(file_index) and index_rate != 0:
            try:
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except Exception as e:
                print(f"[ERROR] Error occurred while reading index file: {e}")
                index = big_npy = None
        else: 
            index = big_npy = None

        opt_ts, audio_opt = [], []
        
        # Handle empty or very short audio
        if len(audio) == 0:
            return np.array([], dtype=np.float32)
        
        audio = signal.filtfilt(bh, ah, audio)
        
        # Skip processing if audio is too short after filtering
        if len(audio) < 320:  # 20ms at 16kHz
            return audio
        
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")

        if audio_pad.shape[0] > self.t_max:
            # Use cumulative sum for efficient windowed sum computation
            audio_abs = np.abs(audio_pad)
            audio_cumsum = np.concatenate([[0], np.cumsum(audio_abs)])
            
            # Compute windowed sum efficiently using cumulative sum
            # Window covers indices [i, i+window)
            audio_window_sum = audio_cumsum[self.window:] - audio_cumsum[:-self.window]
            
            # For each position t, we want the sum of audio in the window around t
            # Create an array where each position has its window sum
            # Extend audio_window_sum to match audio length by padding
            pad_size = self.t_query
            audio_window_sum_padded = np.concatenate([
                audio_window_sum[:pad_size][::-1],  # Mirror left side
                audio_window_sum,
                audio_window_sum[-pad_size:][::-1]   # Mirror right side
            ])
            
            for t in range(self.t_center, audio.shape[0], self.t_center):
                query_start = t
                query_end = t + 2 * self.t_query
                window_sums = audio_window_sum_padded[query_start:query_end]
                if len(window_sums) > 0:
                    min_idx = np.argmin(np.abs(window_sums))
                    opt_ts.append(t + min_idx)

        s = 0
        t = None
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        p_len = audio_pad.shape[0] // self.window

        if pitch_guidance:
            if not hasattr(self, "f0_generator"): 
                self.f0_generator = Generator(self.sample_rate, hop_length, self.f0_min, self.f0_max, self.is_half, self.device, config=self.config)
            pitch, pitchf = self.f0_generator.calculator(f0_method, audio_pad, f0_up_key, p_len, filter_radius, f0_autotune, f0_autotune_strength, proposal_pitch=proposal_pitch, proposal_pitch_threshold=proposal_pitch_threshold)

            if self.device == "mps": 
                pitchf = pitchf.astype(np.float32)
            pitch, pitchf = torch.tensor(pitch[:p_len], device=self.device).unsqueeze(0).long(), torch.tensor(pitchf[:p_len], device=self.device).unsqueeze(0).float()

        if energy_use:
            if not hasattr(self, "rms_extract"): 
                self.rms_extract = RMSEnergyExtractor(frame_length=2048, hop_length=self.window, center=True, pad_mode="reflect").to(self.device).eval()
            energy = self.rms_extract(torch.from_numpy(audio_pad).to(self.device).unsqueeze(0)).cpu().numpy()
            
            if self.device == "mps": 
                energy = energy.astype(np.float32)
            energy = torch.tensor(energy[:p_len], device=self.device).unsqueeze(0).float()

        for t in opt_ts:
            t = t // self.window * self.window
            if t - s < 160:  # Skip very short segments
                continue
                
            segment_audio = audio_pad[s : t + self.t_pad2 + self.window]
            if len(segment_audio) < 320:  # Skip segments that are too short
                continue
                
            audio_opt.append(
                self.voice_conversion(
                    model, 
                    net_g, 
                    sid, 
                    segment_audio, 
                    pitch[:, s // self.window : (t + self.t_pad2) // self.window] if pitch_guidance else None, 
                    pitchf[:, s // self.window : (t + self.t_pad2) // self.window] if pitch_guidance else None, 
                    index, 
                    big_npy, 
                    index_rate, 
                    version, 
                    protect, 
                    energy[:, s // self.window : (t + self.t_pad2) // self.window] if energy_use else None
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )    
            s = t
            
        # Process the last segment
        if s < len(audio_pad):
            segment_audio = audio_pad[s:]
            if len(segment_audio) >= 320:  # Only process if segment is long enough
                audio_opt.append(
                    self.voice_conversion(
                        model, 
                        net_g, 
                        sid, 
                        segment_audio, 
                        (pitch[:, s // self.window :] if t is not None else pitch) if pitch_guidance else None, 
                        (pitchf[:, s // self.window :] if t is not None else pitchf) if pitch_guidance else None, 
                        index, 
                        big_npy, 
                        index_rate, 
                        version, 
                        protect, 
                        (energy[:, s // self.window :] if t is not None else energy) if energy_use else None
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )

        if not audio_opt:
            return np.array([], dtype=np.float32)
            
        audio_opt = np.concatenate(audio_opt)

        if volume_envelope != 1: 
            audio_opt = change_rms(audio, self.sample_rate, audio_opt, self.sample_rate, volume_envelope)
        audio_max = np.abs(audio_opt).max() / 0.99
        if audio_max > 1: 
            audio_opt /= audio_max

        if pitch_guidance: 
            del pitch, pitchf
        del sid

        clear_gpu_cache()
        return audio_opt
