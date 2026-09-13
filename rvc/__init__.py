"""
RVC CLI Tool - A command-line interface for RVC voice conversion
"""

__version__ = "1.0.0"
__author__ = "BF667"
__email__ = ""

from infer.cli import main, convert_audio, VoiceConverter
from infer.infer import infer_main 

__all__ = ['main', 'convert_audio', 'VoiceConverter', 'run_inference_script']
