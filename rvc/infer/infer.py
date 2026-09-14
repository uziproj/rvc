import os
import sys
import torch
import librosa
import logging
import warnings

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")

# Import RVC modules - these will work when the package is installed via pip
from rvc.lib.embedders import fairseq
from rvc.tools.cut import cut, restore
from rvc.infer.pipeline import Pipeline
from rvc.utils import clear_gpu_cache, check_predictors, check_embedders, load_audio
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.lib.config import Config

# Configure logging to silence noisy libraries
for l in ["torch", "faiss", "omegaconf", "httpx", "httpcore", "faiss.loader", "numba.core", "urllib3", "transformers", "matplotlib"]:
    logging.getLogger(l).setLevel(logging.ERROR)


# Supported audio extensions for batch conversion.
_AUDIO_EXTS = ("wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4",
               "aac", "alac", "wma", "aiff", "webm", "ac3")


class RVClass:
    """High-level RVC inference interface.

    Wraps the lower-level :class:`VoiceConverter` so user code can stay short:

        from rvc import Config, RVClass

        config = Config(embedder_model="contentvec_base", f0_method="rmvpe")
        rvc = RVClass(config=config, pth_path="model.pth")

        # single file
        rvc.convert(input_path="in.wav", output_path="out.wav", pitch=12)

        # or auto-detect (batch if input is a directory)
        rvc.run(input_path="in.wav", output_path="out.wav", pitch=12)
        rvc.run(input_path="./audio_folder", pitch=12)

    The model is loaded once at ``__init__`` and reused for every call to
    :meth:`convert`, :meth:`convert_batch`, or :meth:`run`.
    """

    def __init__(
        self,
        config,
        pth_path,
        sid=0,
        embedder_model="contentvec_base",
        f0_method="rmvpe",
    ):
        """Initialize the converter and load the voice model.

        Args:
            config: A :class:`rvc.lib.config.Config` instance. Hubert and
                RMVPE models are expected to be preloaded there.
            pth_path: Path to the ``.pth`` voice model file.
            sid: Speaker ID (for multi-speaker models).
            embedder_model: Embedder model name (used for predictor checks).
            f0_method: F0 method name (used for predictor checks).
        """
        check_predictors(f0_method)
        check_embedders(embedder_model)

        if not pth_path or not os.path.exists(pth_path) \
                or os.path.isdir(pth_path) or not pth_path.endswith(".pth"):
            raise FileNotFoundError(
                f"[ERROR] Please enter a valid model path (.pth): {pth_path!r}"
            )

        self.config = config
        self.pth_path = pth_path
        self.sid = sid
        self.embedder_model = embedder_model
        self.f0_method = f0_method

        # VoiceConverter handles the actual model loading + inference.
        self._converter = VoiceConverter(config, pth_path, sid)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def convert(
        self,
        input_path,
        output_path="./output.wav",
        pitch=0,
        filter_radius=3,
        index_rate=0.5,
        volume_envelope=1,
        protect=0.5,
        hop_length=64,
        f0_method="rmvpe",
        index_path=None,
        export_format="wav",
        embedder_model="contentvec_base",
        resample_sr=0,
        f0_autotune=False,
        f0_autotune_strength=1,
        split_audio=False,
        clean_audio=False,
        clean_strength=0.7,
        formant_shifting=False,
        formant_qfrency=0.8,
        formant_timbre=0.8,
        proposal_pitch=False,
        proposal_pitch_threshold=255.0,
    ):
        """Convert a single audio file."""
        if not os.path.exists(input_path):
            print("[WARNING] No audio files found.")
            return False

        check_predictors(f0_method)
        check_embedders(embedder_model)

        print(f"[INFO] Conversion '{input_path}'...")
        if os.path.exists(output_path):
            os.remove(output_path)

        self._converter.convert_audio(
            audio_input_path=input_path,
            audio_output_path=output_path,
            index_path=index_path,
            embedder_model=embedder_model,
            pitch=pitch,
            f0_method=f0_method,
            index_rate=index_rate,
            volume_envelope=volume_envelope,
            protect=protect,
            hop_length=hop_length,
            filter_radius=filter_radius,
            export_format=export_format,
            resample_sr=resample_sr,
            f0_autotune=f0_autotune,
            f0_autotune_strength=f0_autotune_strength,
            split_audio=split_audio,
            clean_audio=clean_audio,
            clean_strength=clean_strength,
            formant_shifting=formant_shifting,
            formant_qfrency=formant_qfrency,
            formant_timbre=formant_timbre,
            proposal_pitch=proposal_pitch,
            proposal_pitch_threshold=proposal_pitch_threshold,
        )

        print("[INFO] Conversion complete.")
        return True

    def convert_batch(
        self,
        input_dir,
        pitch=0,
        filter_radius=3,
        index_rate=0.5,
        volume_envelope=1,
        protect=0.5,
        hop_length=64,
        f0_method="rmvpe",
        index_path=None,
        export_format="wav",
        embedder_model="contentvec_base",
        resample_sr=0,
        f0_autotune=False,
        f0_autotune_strength=1,
        split_audio=False,
        clean_audio=False,
        clean_strength=0.7,
        formant_shifting=False,
        formant_qfrency=0.8,
        formant_timbre=0.8,
        proposal_pitch=False,
        proposal_pitch_threshold=255.0,
    ):
        """Batch-convert every audio file in a directory."""
        print("[INFO] Use batch conversion...")
        audio_files = [
            f for f in os.listdir(input_dir)
            if f.lower().endswith(_AUDIO_EXTS)
        ]

        if not audio_files:
            print("[WARNING] No audio files found.")
            return False

        print(f"[INFO] Found {len(audio_files)} audio files for conversion.")

        check_predictors(f0_method)
        check_embedders(embedder_model)

        for audio in audio_files:
            audio_path = os.path.join(input_dir, audio)
            output_audio = os.path.join(
                input_dir,
                os.path.splitext(audio)[0] + f"_output.{export_format}",
            )

            print(f"[INFO] Conversion '{audio_path}'...")
            if os.path.exists(output_audio):
                os.remove(output_audio)

            self._converter.convert_audio(
                audio_input_path=audio_path,
                audio_output_path=output_audio,
                index_path=index_path,
                embedder_model=embedder_model,
                pitch=pitch,
                f0_method=f0_method,
                index_rate=index_rate,
                volume_envelope=volume_envelope,
                protect=protect,
                hop_length=hop_length,
                filter_radius=filter_radius,
                export_format=export_format,
                resample_sr=resample_sr,
                f0_autotune=f0_autotune,
                f0_autotune_strength=f0_autotune_strength,
                split_audio=split_audio,
                clean_audio=clean_audio,
                clean_strength=clean_strength,
                formant_shifting=formant_shifting,
                formant_qfrency=formant_qfrency,
                formant_timbre=formant_timbre,
                proposal_pitch=proposal_pitch,
                proposal_pitch_threshold=proposal_pitch_threshold,
            )

        print("[INFO] Conversion complete.")
        return True

    def run(self, input_path, output_path="./output.wav", **kwargs):
        """Auto-dispatch: batch if ``input_path`` is a directory, else single."""
        if os.path.isdir(input_path):
            # batch mode ignores output_path (per-file outputs go into input dir)
            return self.convert_batch(input_dir=input_path, **kwargs)
        return self.convert(input_path=input_path, output_path=output_path, **kwargs)

    # ------------------------------------------------------------------
    # Cleanup / context-manager support
    # ------------------------------------------------------------------
    def cleanup(self):
        """Release GPU memory and clear internal state."""
        self._converter.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False


# ---------------------------------------------------------------------------
# Backwards-compatible functional API
# ---------------------------------------------------------------------------
def infer_main(
    config,
    pitch=0,
    filter_radius=3,
    index_rate=0.5,
    volume_envelope=1,
    protect=0.5,
    hop_length=64,
    f0_method="rmvpe",
    input_path=None,
    output_path="./output.wav",
    pth_path=None,
    index_path=None,
    export_format="wav",
    embedder_model="contentvec_base",
    resample_sr=0,
    f0_autotune=False,
    f0_autotune_strength=1,
    split_audio=False,
    clean_audio=False,
    clean_strength=0.7,
    formant_shifting=False,
    formant_qfrency=0.8,
    formant_timbre=0.8,
    proposal_pitch=False,
    proposal_pitch_threshold=255.0,
):
    """One-shot inference entry point (function-style wrapper around RVClass).

    Maintained for backwards compatibility — new code should prefer
    instantiating :class:`RVClass` directly so the model is loaded once
    and reused across multiple conversions.
    """
    rvc = RVClass(
        config=config,
        pth_path=pth_path,
        sid=0,
        embedder_model=embedder_model,
        f0_method=f0_method,
    )

    return rvc.run(
        input_path=input_path,
        output_path=output_path,
        pitch=pitch,
        filter_radius=filter_radius,
        index_rate=index_rate,
        volume_envelope=volume_envelope,
        protect=protect,
        hop_length=hop_length,
        f0_method=f0_method,
        index_path=index_path,
        export_format=export_format,
        embedder_model=embedder_model,
        resample_sr=resample_sr,
        f0_autotune=f0_autotune,
        f0_autotune_strength=f0_autotune_strength,
        split_audio=split_audio,
        clean_audio=clean_audio,
        clean_strength=clean_strength,
        formant_shifting=formant_shifting,
        formant_qfrency=formant_qfrency,
        formant_timbre=formant_timbre,
        proposal_pitch=proposal_pitch,
        proposal_pitch_threshold=proposal_pitch_threshold,
    )


# Backwards-compatible alias: README / DOCUMENTATION / Colab notebook
# import `run_inference_script` — keep both names pointing at the same callable.
run_inference_script = infer_main


class VoiceConverter:
    def __init__(self, config, model_path, sid = 0):
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
        resample_sr = 0,
        f0_autotune=False,
        f0_autotune_strength=1,
        split_audio=False,
        clean_audio=False,
        clean_strength=0.7,
        formant_shifting=False,
        formant_qfrency=0.8,
        formant_timbre=0.8,
        proposal_pitch=False,
        proposal_pitch_threshold=255.0
    ):
        try:
            audio = load_audio(audio_input_path, self.sample_rate, formant_shifting=formant_shifting, formant_qfrency=formant_qfrency, formant_timbre=formant_timbre)
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1: audio /= audio_max

            if not self.hubert_model:
                if self.config.hubert_model is not None:
                    self.hubert_model = self.config.hubert_model
                else:
                    check_embedders(embedder_model)
                    embedder_model_path = os.path.join(os.getcwd(), "assets", "models", embedder_model + ".pt")
                    if not os.path.exists(embedder_model_path): raise FileNotFoundError(f"[ERROR] Not found embedder: {embedder_model}")

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
            else: chunks = [(audio, 0, 0)]

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
            if self.cpt is not None: self.setup()

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
        if os.path.isfile(self.loaded_model): self.cpt = torch.load(self.loaded_model, map_location="cpu")
        else: self.cpt = None

    def setup(self):
        if self.cpt is not None:
            self.tgt_sr = self.cpt["config"][-1]
            self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]

            self.use_f0 = self.cpt.get("f0", 1)
            self.version = self.cpt.get("version", "v1")
            self.vocoder = self.cpt.get("vocoder", "Default")
            self.energy = self.cpt.get("energy", False)

            if self.vocoder != "Default": self.config.is_half = False
            self.net_g = Synthesizer(*self.cpt["config"], use_f0=self.use_f0, text_enc_hidden_dim=768 if self.version == "v2" else 256, vocoder=self.vocoder, energy=self.energy)
            del self.net_g.enc_q

            self.net_g.load_state_dict(self.cpt["weight"], strict=False)
            self.net_g.eval().to(self.device)
            self.net_g = (self.net_g.half() if self.config.is_half else self.net_g.float())
            self.n_spk = self.cpt["config"][-3]

            self.vc = Pipeline(self.tgt_sr, self.config)
