"""RVC inference sub-package.

Exposes the public Python API entry points (`infer_main`,
`run_inference_script`) so users can import either name:

    from rvc.infer import run_inference_script
    from rvc.infer.infer import infer_main
"""

from rvc.infer.infer import infer_main, VoiceConverter

# Backwards-compatible alias kept in sync with the top-level package alias.
run_inference_script = infer_main

__all__ = ["infer_main", "run_inference_script", "VoiceConverter"]
