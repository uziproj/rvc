# RVC Examples

Example scripts showing how to use the RVC Python API.

## Files

| File | What it shows |
|------|---------------|
| [`infer.py`](./infer.py) | Basic single-file conversion using `RVClass` (recommended API) |
| [`batch.py`](./batch.py) | Batch-convert every audio file in a directory |
| [`multi_pitch.py`](./multi_pitch.py) | Convert the same audio with multiple pitch shifts (one model load, many runs) |
| [`compare_f0.py`](./compare_f0.py) | A/B test different F0 extraction methods on the same input |
| [`function_api.py`](./function_api.py) | Use the backwards-compatible `run_inference_script()` function |
| [`logging_levels.py`](./logging_levels.py) | Control log verbosity via `Config`, `RVClass`, and `set_log_level()` |
| [`api_client.py`](./api_client.py) | Talk to the REST API server (`rvc-api`) from Python using `requests` |

## Quick start

```bash
pip install git+https://github.com/uziproj/rvc.git
```

Then run any example:

```bash
# Single file (recommended)
python examples/infer.py input.wav model.pth output.wav --pitch 12

# Batch
python examples/batch.py ./audio_folder model.pth --pitch 12 --output-format flac

# Multiple pitch shifts (same model)
python examples/multi_pitch.py input.wav model.pth ./outputs --pitches -12 0 12

# Compare F0 methods on the same input
python examples/compare_f0.py input.wav model.pth ./outputs --methods rmvpe fcpe crepe-large

# Function-style API (backwards-compatible)
python examples/function_api.py input.wav model.pth output.wav --pitch 12

# Logging level demo
python examples/logging_levels.py input.wav model.pth

# REST API client (requires server running: rvc-api --port 8000)
python examples/api_client.py input.wav /path/to/model.pth output.wav --pitch 12
```

## The recommended pattern

```python
from rvc import Config, RVClass

# Set f0_method + embedder_model ONCE on Config — single source of truth
config = Config(embedder_model="contentvec_base", f0_method="rmvpe")

# RVClass inherits them — no need to repeat
with RVClass(config=config, pth_path="model.pth") as rvc:
    # Only per-conversion params here
    rvc.run(input_path="in.wav", output_path="out.wav", pitch=12)

    # Batch (auto-detected from directory input)
    rvc.run(input_path="./audio_folder", pitch=12)

    # Override f0_method for a single call only
    rvc.run(input_path="tricky.wav", f0_method="crepe-large", pitch=12)
```

See [`infer.py`](./infer.py) for the simplest end-to-end example.
