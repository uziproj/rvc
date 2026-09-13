"""
RVC CLI Tool - A command-line interface for RVC voice conversion
"""

__version__ = "0.1.0"
__author__ = "uziproj"
__email__ = ""

from rvc.infer.cli import main, convert_audio, VoiceConverter
from rvc.infer.infer import infer_main

# Backwards-compatible alias: the README / DOCUMENTATION / Colab notebook
# all reference `run_inference_script`. Keep the alias so external code
# continues to work after this fix.
run_inference_script = infer_main

__all__ = [
    "main",
    "convert_audio",
    "VoiceConverter",
    "infer_main",
    "run_inference_script",
    "__version__",
    "__author__",
]
