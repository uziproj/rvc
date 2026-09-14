"""
RVC - Retrieval-based Voice Conversion

A simple, high-quality voice conversion tool.

Top-level package that re-exports the most common entry points so users
can get started with a single import line:

    from rvc import *

    config = Config(embedder_model="contentvec_base", f0_method="rmvpe")
    run_inference_script(
        config=config,
        input_path="input.wav",
        output_path="output.wav",
        pth_path="model.pth",
        pitch=12,
        f0_method="rmvpe",
    )

Or pick names explicitly:

    from rvc import Config, run_inference_script, F0_METHODS, VoiceConverter

For the REST API server, import from the sub-package directly (this avoids
pulling FastAPI into every `import rvc` call):

    from rvc.api.app import app, main as api_main
"""

__version__ = "0.1.0"
__author__ = "uziproj"
__email__ = ""

# --- CLI entry points ----------------------------------------------------
from rvc.infer.cli import main, convert_audio, VoiceConverter

# --- Python API ----------------------------------------------------------
from rvc.infer.infer import infer_main, RVClass

# Backwards-compatible alias: README / DOCUMENTATION / Colab notebook
# all reference `run_inference_script`. Both names resolve to the same
# callable; existing scripts that used `infer_main` keep working.
run_inference_script = infer_main

# --- Configuration & inference building blocks ---------------------------
from rvc.lib.config import Config, PREDICTOR_MODEL
from rvc.infer.pipeline import Pipeline
from rvc.lib.predictor.generator import Generator
from rvc.utils import (
    Autotune,
    load_audio,
    check_predictors,
    check_embedders,
    clear_gpu_cache,
    change_rms,
    HF_download_file,
)

# --- F0 methods list -----------------------------------------------------
from rvc.var import method as F0_METHODS

__all__ = [
    # Meta
    "__version__",
    "__author__",
    "__email__",
    # CLI
    "main",
    "convert_audio",
    "VoiceConverter",
    # Python API
    "infer_main",
    "run_inference_script",
    "RVClass",
    # Config & inference
    "Config",
    "PREDICTOR_MODEL",
    "Pipeline",
    "Generator",
    "Autotune",
    "load_audio",
    "check_predictors",
    "check_embedders",
    "clear_gpu_cache",
    "change_rms",
    "HF_download_file",
    # F0
    "F0_METHODS",
]
