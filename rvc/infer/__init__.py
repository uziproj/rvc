"""RVC inference sub-package.

Exposes the public Python API entry points (`RVClass`, `infer_main`,
`run_inference_script`) so users can import either name:

    # recommended (class form)
    from rvc import Config, RVClass

    rvc = RVClass(config=Config(), pth_path="model.pth")
    rvc.run("in.wav", "out.wav", pitch=12)

    # or function form (kept for backwards compatibility)
    from rvc.infer import run_inference_script
    from rvc.infer.infer import infer_main
"""

from rvc.infer.infer import infer_main, run_inference_script, RVClass, VoiceConverter

__all__ = ["RVClass", "infer_main", "run_inference_script", "VoiceConverter"]
