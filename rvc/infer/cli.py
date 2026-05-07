#!/usr/bin/env python3
"""
RVC Voice Conversion CLI Tool
"""

import os
import sys
import argparse
import torch
import librosa
import logging
import warnings

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")

# Import RVC modules - these will work when the package is installed via pip
# No need for sys.path manipulation as the package will be discoverable
try:
    from rvc.lib.embedders import fairseq
    from rvc.tools.cut import cut, restore
    from rvc.infer.pipeline import Pipeline
    from rvc.utils import clear_gpu_cache, check_predictors, check_embedders, load_audio
    from rvc.lib.algorithm.synthesizers import Synthesizer
    from rvc.lib.config import Config
except ImportError as e:
    print(f"[ERROR] Required RVC modules not found: {e}")
    print("Please install the RVC library first: pip install .")
    sys.exit(1)
from rvc.var import method
# Configure logging to silence noisy libraries
for l in ["torch", "faiss", "omegaconf", "httpx", "httpcore", "faiss.loader", "numba.core", "urllib3", "transformers", "matplotlib"]:
    logging.getLogger(l).setLevel(logging.ERROR)


class VoiceConverter:
    def __init__(self, config, model_path, sid=0):
        self.config = config
        self.device = config.device
        self.hubert_model = None
        self.tgt_sr = None 
        self.net_g = None 
        self.vc = None
        self.cpt = None  
        self.version = None 
        self.n_spk = None  
        self.use_f0 = None  
        self.loaded_model = None
        self.vocoder = "Default"
        self.sample_rate = 16000
        self.sid = sid
        self.get_vc(model_path, sid)

    def convert_audio(
        self, 
        audio_input_path, 
        audio_output_path, 
        index_path, 
        embedder_model, 
        pitch, 
        f0_method, 
        index_rate, 
        volume_envelope, 
        protect, 
        hop_length, 
        filter_radius, 
        export_format, 
        resample_sr=0, 
        f0_autotune=False, 
        f0_autotune_strength=1,
        split_audio=False,
        clean_audio=False,
        clean_strength=0.5,
        formant_shifting=False,
        formant_qfrency=0.8, 
        formant_timbre=0.8, 
        proposal_pitch=False, 
        proposal_pitch_threshold=255.0
    ):
        try:
            check_predictors(f0_method)
            audio = load_audio(audio_input_path, self.sample_rate, formant_shifting=formant_shifting, formant_qfrency=formant_qfrency, formant_timbre=formant_timbre)
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max

            if not self.hubert_model:
                if self.config.hubert_model is not None:
                    self.hubert_model = self.config.hubert_model
                else:
                    check_embedders(embedder_model)
                    HUBERT_PATH = os.path.join(os.getcwd(), "assets", "models")
                    embedder_model_path = os.path.join(HUBERT_PATH, embedder_model + ".pt")
                    if not os.path.exists(embedder_model_path):
                        raise FileNotFoundError(f"[ERROR] Not found embeddeder: {embedder_model}")

                    models = fairseq.load_model(embedder_model_path).to(self.device).eval()
                    self.hubert_model = models.half() if self.config.is_half else models.float()

            if split_audio:
                chunks = cut(
                    audio, 
                    self.sample_rate, 
                    db_thresh=-60, 
                    min_interval=500
                )  
                print(f"[INFO] Split Total: {len(chunks)}")
            else:
                chunks = [(audio, 0, 0)]

            converted_chunks = [
                (
                    start, 
                    end, 
                    self.vc.pipeline(
                        model=self.hubert_model, 
                        net_g=self.net_g, 
                        sid=self.sid, 
                        audio=waveform, 
                        f0_up_key=pitch, 
                        f0_method=f0_method, 
                        file_index=(
                            index_path.strip().strip('"').strip("\n").strip('"').strip().replace("trained", "added")
                        ), 
                        index_rate=index_rate, 
                        pitch_guidance=self.use_f0, 
                        filter_radius=filter_radius, 
                        volume_envelope=volume_envelope, 
                        version=self.version, 
                        protect=protect, 
                        hop_length=hop_length, 
                        energy_use=self.energy,
                        f0_autotune=f0_autotune, 
                        f0_autotune_strength=f0_autotune_strength,
                        proposal_pitch=proposal_pitch,
                        proposal_pitch_threshold=proposal_pitch_threshold
                    )
                ) for waveform, start, end in chunks
            ]

            audio_output = restore(
                converted_chunks, 
                total_len=len(audio), 
                dtype=converted_chunks[0][2].dtype
            ) if split_audio else converted_chunks[0][2]

            if self.tgt_sr != resample_sr and resample_sr > 0: 
                audio_output = librosa.resample(audio_output, orig_sr=self.tgt_sr, target_sr=resample_sr, res_type="soxr_vhq")
                self.tgt_sr = resample_sr

            if clean_audio:
                from rvc.tools.noisereduce import reduce_noise
                audio_output = reduce_noise(
                    y=audio_output, 
                    sr=self.tgt_sr, 
                    prop_decrease=clean_strength, 
                    device=self.device
                ) 

            sf.write(audio_output_path, audio_output, self.tgt_sr, format=export_format)
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            print(f"[ERROR] An error has occurred: {e}")

    def get_vc(self, weight_root, sid):
        if sid == "" or sid == []:
            self.cleanup()
            clear_gpu_cache()

        if not self.loaded_model or self.loaded_model != weight_root:
            self.loaded_model = weight_root
            self.load_model()
            if self.cpt is not None:
                self.setup()

    def cleanup(self):
        if self.hubert_model is not None:
            del self.net_g, self.n_spk, self.vc, self.hubert_model, self.tgt_sr
            self.hubert_model = self.net_g = self.n_spk = self.vc = self.tgt_sr = None
            clear_gpu_cache()
        if hasattr(self, 'cpt'):
            del self.cpt
            self.cpt = None
        clear_gpu_cache()

    def load_model(self):
        if os.path.isfile(self.loaded_model):
            self.cpt = torch.load(self.loaded_model, map_location="cpu")  
        else:
            self.cpt = None

    def setup(self):
        if self.cpt is not None:
            self.tgt_sr = self.cpt["config"][-1]
            self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]

            self.use_f0 = self.cpt.get("f0", 1)
            self.version = self.cpt.get("version", "v1")
            self.vocoder = self.cpt.get("vocoder", "Default")
            self.energy = self.cpt.get("energy", False)

            if self.vocoder != "Default":
                self.config.is_half = False
            self.net_g = Synthesizer(*self.cpt["config"], use_f0=self.use_f0, text_enc_hidden_dim=768 if self.version == "v2" else 256, vocoder=self.vocoder, energy=self.energy)
            del self.net_g.enc_q

            self.net_g.load_state_dict(self.cpt["weight"], strict=False)
            self.net_g.eval().to(self.device)
            self.net_g = (self.net_g.half() if self.config.is_half else self.net_g.float())
            self.n_spk = self.cpt["config"][-3]

            self.vc = Pipeline(self.tgt_sr, self.config)


def convert_audio(args):
    """Main conversion function"""
    config = Config(embedder_model=args.embedder, f0_method=args.f0_method)

    check_predictors(args.f0_method)
    check_embedders(args.embedder)

    # Model validation
    if not args.model or not os.path.exists(args.model) or os.path.isdir(args.model) or not args.model.endswith(".pth"):
        print("[ERROR] Please enter a valid .pth model file.")
        return False

    cvt = VoiceConverter(config, args.model, 0)

    input_path = args.input
    output_path = args.output

    if os.path.isdir(input_path):
        print("[INFO] Batch conversion mode...")
        audio_files = [f for f in os.listdir(input_path) if f.lower().endswith(("wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4", "aac", "alac", "wma", "aiff", "webm", "ac3"))]

        if not audio_files: 
            print("[WARNING] No audio files found.")
            return False

        print(f"[INFO] Found {len(audio_files)} audio files for conversion.")

        for audio in audio_files:
            audio_path = os.path.join(input_path, audio)
            output_audio = os.path.join(input_path, os.path.splitext(audio)[0] + f"_output.{args.format}")

            print(f"[INFO] Converting '{audio_path}'...")
            if os.path.exists(output_audio):
                os.remove(output_audio)

            cvt.convert_audio(
                audio_input_path=audio_path, 
                audio_output_path=output_audio, 
                index_path=args.index, 
                embedder_model=args.embedder, 
                pitch=args.pitch, 
                f0_method=args.f0_method, 
                index_rate=args.index_rate, 
                volume_envelope=args.volume_envelope, 
                protect=args.protect, 
                hop_length=args.hop_length, 
                filter_radius=args.filter_radius, 
                export_format=args.format, 
                resample_sr=args.resample_sr, 
                f0_autotune=args.f0_autotune, 
                f0_autotune_strength=args.f0_autotune_strength,
                split_audio=args.split_audio,
                clean_audio=args.clean_audio,
                clean_strength=args.clean_strength,
                formant_shifting=args.formant_shifting,
                formant_qfrency=args.formant_qfrency, 
                formant_timbre=args.formant_timbre,
                proposal_pitch=args.proposal_pitch,
                proposal_pitch_threshold=args.proposal_pitch_threshold
            )

        print("[INFO] Batch conversion complete.")
        return True
    else:
        if not os.path.exists(input_path):
            print("[ERROR] Input file not found.")
            return False

        print(f"[INFO] Converting '{input_path}'...")
        if os.path.exists(output_path):
            os.remove(output_path)

        cvt.convert_audio(
            audio_input_path=input_path, 
            audio_output_path=output_path, 
            index_path=args.index, 
            embedder_model=args.embedder, 
            pitch=args.pitch, 
            f0_method=args.f0_method, 
            index_rate=args.index_rate, 
            volume_envelope=args.volume_envelope, 
            protect=args.protect, 
            hop_length=args.hop_length, 
            filter_radius=args.filter_radius,  
            export_format=args.format, 
            resample_sr=args.resample_sr, 
            f0_autotune=args.f0_autotune, 
            f0_autotune_strength=args.f0_autotune_strength,
            split_audio=args.split_audio,
            clean_audio=args.clean_audio,
            clean_strength=args.clean_strength,
            formant_shifting=args.formant_shifting,
            formant_qfrency=args.formant_qfrency, 
            formant_timbre=args.formant_timbre,
            proposal_pitch=args.proposal_pitch,
            proposal_pitch_threshold=args.proposal_pitch_threshold
        )

        print(f"[INFO] Conversion complete. Output saved to: {output_path}")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="RVC Voice Conversion CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  rvc-cli -i input.wav -o output.wav -m model.pth
  rvc-cli -i ./audio_folder -m model.pth -p 12 -f0 rmvpe
  rvc-cli -i input.wav -m model.pth -idx model.index -ir 0.75 -clean
        """
    )

    # Required Arguments
    parser.add_argument("-i", "--input", type=str, required=True,
                       help="Path to input audio file or directory")
    parser.add_argument("-o", "--output", type=str, default="./output.wav",
                       help="Path to output audio file")
    parser.add_argument("-m", "--model", type=str, required=True,
                       help="Path to .pth model file")

    # Optional Arguments - Model & Processing
    parser.add_argument("-idx", "--index", type=str, default=None,
                       help="Path to .index file")
    parser.add_argument("-em", "--embedder", type=str, default="contentvec_base",
                       help="Embedder model (contentvec_base, hubert_base)")
    parser.add_argument("-f0", "--f0_method", type=str, default="rmvpe",
                        help=f"F0 prediction method, available model {method}")
    
    # Optional Arguments - Audio Adjustments
    parser.add_argument("-p", "--pitch", type=int, default=0,
                       help="Pitch shift in semitones")
    parser.add_argument("-ir", "--index_rate", type=float, default=0.5,
                       help="Index rate (feature retrieval ratio)")
    parser.add_argument("-ve", "--volume_envelope", type=float, default=1.0,
                       help="Volume envelope ratio")
    parser.add_argument("-pr", "--protect", type=float, default=0.5,
                       help="Protect voiceless consonants")
    parser.add_argument("-fr", "--filter_radius", type=int, default=3,
                       help="Filter radius")
    parser.add_argument("-hl", "--hop_length", type=int, default=64,
                       help="Hop length")
    parser.add_argument("-rs", "--resample_sr", type=int, default=0,
                       help="Resample sample rate (0 to disable)")
    
    # Optional Arguments - Output & Extras
    parser.add_argument("-fmt", "--format", type=str, default="wav",
                       help="Output format (wav, flac, mp3, ogg)")
    parser.add_argument("-split", "--split_audio", action="store_true",
                       help="Split audio into chunks for processing")
    parser.add_argument("-clean", "--clean_audio", action="store_true",
                       help="Apply noise reduction to output")
    parser.add_argument("-cs", "--clean_strength", type=float, default=0.7,
                       help="Noise reduction strength")

    # Optional Arguments - Tuning
    parser.add_argument("-fa", "--f0_autotune", action="store_true",
                       help="Enable F0 autotune")
    parser.add_argument("-fas", "--f0_autotune_strength", type=float, default=1.0,
                       help="F0 autotune strength")
    
    parser.add_argument("-fs", "--formant_shifting", action="store_true",
                       help="Enable formant shifting")
    parser.add_argument("-fq", "--formant_qfrency", type=float, default=0.8,
                       help="Formant quefrency")
    parser.add_argument("-ft", "--formant_timbre", type=float, default=0.8,
                       help="Formant timbre")

    parser.add_argument("-pp", "--proposal_pitch", action="store_true",
                       help="Enable proposal pitch")
    parser.add_argument("-ppt", "--proposal_pitch_threshold", type=float, default=255.0,
                       help="Proposal pitch threshold")

    # Version and info
    parser.add_argument("-v", "--version", action="store_true",
                       help="Show version information")
    
    args = parser.parse_args()

    if args.version:
        from rvc.infer import __version__
        print(f"RVC CLI Tool v{__version__}")
        return

    success = convert_audio(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
