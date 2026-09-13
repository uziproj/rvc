<div align="center">

# Simple RVC

A simple, high-quality voice conversion tool focused on simplicity and ease of use.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uziproj/rvc/blob/main/colab/rvc_demo.ipynb)
[![Python 3.10-3.12](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>


## Key Features

* **20+ pitch extraction methods**: pm, dio, crepe, fcpe, rmvpe, harvest, yin, pyin, swipe, djcm and more
* **Hybrid f0 methods**: combine multiple pitch extractors (e.g. `hybrid[rmvpe+fcpe]`) for robust results
* **Powerful CLI** for single-file and batch processing
* **Multiple embedder models**: contentvec, hubert (multilingual), spin
* **Advanced features**: formant shifting, noise reduction, autotune with adjustable strength, proposal pitch
* **REST API**: FastAPI-based HTTP server for integration into any application
* **Eager model loading**: Hubert and RMVPE models preload at startup for faster first inference
* **ONNX export** for optimized inference
* **Multi-backend support**: NVIDIA CUDA, AMD OpenCL, Apple MPS, CPU fallback

## Quick Start

### Try it in Google Colab
Click the badge below to open a ready-to-run demo notebook — no local installation required:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uziproj/rvc/blob/main/colab/rvc_demo.ipynb)

### Install

```bash
pip install git+https://github.com/uziproj/rvc.git
```

### CLI

```bash
rvc -i input.wav -o output.wav -m model.pth -p 12
```

### Python API

All public symbols are re-exported at the top level, so `from rvc import *` (or
explicit imports like `from rvc import Config, run_inference_script`) is the
recommended way to use the library.

```python
from rvc import Config, run_inference_script

# Hubert and RMVPE models load eagerly at Config initialization
config = Config(embedder_model="contentvec_base", f0_method="rmvpe")

run_inference_script(
    config=config,
    input_path="input.wav",
    output_path="output.wav",
    pth_path="model.pth",
    pitch=12,
    f0_method="rmvpe",
)
```

You can also use the wildcard form, which exposes every public name:

```python
from rvc import *

print(__version__)          # "0.1.0"
print(F0_METHODS[:3])       # ['pm', 'dio', 'mangio-crepe-tiny']

config = Config()
converter = VoiceConverter(config, model_path="model.pth", sid=0)
```

## Installation

**Requires Python 3.10-3.12 and FFmpeg installed**

### Option 1: Install from Git (Recommended)
```bash
pip install git+https://github.com/uziproj/rvc.git
```

### Option 2: Install from Source (Development)
```bash
git clone https://github.com/uziproj/rvc.git
cd rvc
pip install -e .
```

### Python Version Compatibility

This project supports Python 3.10, 3.11, and 3.12. The code has been updated to be compatible with newer numpy versions (2.0+) which resolves compatibility issues that previously existed with Python 3.11+ installations.

### Platform-Specific Notes

**NVIDIA GPU:**
- PyTorch with CUDA support is automatically installed
- Ensure you have CUDA drivers installed

**AMD GPU:**
- May require additional setup for OpenCL support
- Consider using ROCm PyTorch if available for your hardware

**CPU Only:**
- The package will automatically use CPU inference
- Note: CPU inference will be slower than GPU


## Command Line Interface

After installation, you can use the RVC CLI tool for voice conversion:

```bash
rvc -i input.wav -o output.wav -m model.pth
```

**Basic Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `-i, --input` | Path to input audio file or directory | (required) |
| `-o, --output` | Path to output audio file | `./output.wav` |
| `-m, --model` | Path to .pth model file | (required) |

**Common Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `-p, --pitch` | Pitch shift in semitones | `0` |
| `-f0, --f0_method` | F0 prediction method | `rmvpe` |
| `-idx, --index` | Path to .index file | `None` |
| `-ir, --index_rate` | Index rate for feature retrieval | `0.5` |
| `-em, --embedder` | Embedder model name | `contentvec_base` |

**Advanced Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `-split, --split_audio` | Split audio into chunks for processing | `False` |
| `-clean, --clean_audio` | Apply noise reduction to output | `False` |
| `-cs, --clean_strength` | Noise reduction strength (0.0-1.0) | `0.7` |
| `-fa, --f0_autotune` | Enable F0 autotune | `False` |
| `-fas, --f0_autotune_strength` | Autotune strength (0.0-1.0) | `1.0` |
| `-fs, --formant_shifting` | Enable formant shifting | `False` |
| `-fq, --formant_qfrency` | Formant quefrency | `0.8` |
| `-ft, --formant_timbre` | Formant timbre | `0.8` |
| `-pp, --proposal_pitch` | Enable proposal pitch | `False` |
| `-ppt, --proposal_pitch_threshold` | Proposal pitch threshold | `255.0` |
| `-fr, --filter_radius` | Filter radius for pitch extraction | `3` |
| `-hl, --hop_length` | Hop length for pitch extraction | `64` |
| `-rs, --resample_sr` | Resample output sample rate (0=disabled) | `0` |
| `-fmt, --format` | Output format (wav, flac, mp3, ogg) | `wav` |

### Available F0 Methods

| Method | Quality | Speed | Description |
|--------|---------|-------|-------------|
| `pm` | Low | Fastest | Parselmouth (Praat) autocorrelation — Boersma (1993) |
| `dio` | Low | Fast | PyWorld DIO — Morise & Kawahara (2010) |
| `harvest` | Medium | Medium | PyWorld Harvest — Morise (2017) |
| `yin` | Medium | Medium | YIN — de Cheveigné & Kawahara (2002) |
| `pyin` | Medium | Medium | pYIN — Mauch & Dixon (2014) |
| `swipe` | Medium | Medium | SWIPE — Camacho & Harris (2008) |
| `rmvpe` | High | Slow | RMVPE (U-Net) — Wei et al. (2023, Inter Speech) |
| `rmvpe-legacy` | High | Slow | RMVPE with StoneMask pitch refinement |
| `fcpe` | High | Slow | FCPE (Lynx-Net) — CNChTu (2025) |
| `fcpe-legacy` | High | Slow | FCPE with higher voicing threshold |
| `crepe-*` | Varies | Varies | CREPE CNN — Kim et al. (2018) — 5 sizes: tiny/small/medium/large/full |
| `mangio-crepe-*` | Varies | Varies | CREPE (Mangio fork) — no periodicity filter, raw Viterbi output |
| `djcm` | High | Slow | DJCM — Wei et al. (2024, ICASSP) |
| `hybrid[m1+m2]` | High | Slow | Weighted geometric mean of multiple methods |

### Example Usage

```bash
# Simple conversion
rvc -i input.wav -o output.wav -m model.pth -p 12

# Batch conversion from directory
rvc -i ./audio_folder -m model.pth -p 12 -f0 rmvpe

# With index file and autotune
rvc -i input.wav -m model.pth -idx model.index -ir 0.75 -fa

# With noise reduction and formant shifting
rvc -i input.wav -o output.wav -m model.pth -clean -fs -fq 0.9 -ft 0.7

# Using CREPE large for highest quality
rvc -i input.wav -o output.wav -m model.pth -f0 crepe-large

# Hybrid method combining RMVPE and FCPE
rvc -i input.wav -o output.wav -m model.pth -f0 "hybrid[rmvpe+fcpe]"
```

For more options, run:
```bash
rvc --help
```

## REST API

RVC includes a built-in REST API server powered by FastAPI, allowing you to integrate voice conversion into any application over HTTP.

### Starting the Server

```bash
# Using the CLI entry point
rvc-api --host 0.0.0.0 --port 8000

# Or with uvicorn directly
uvicorn rvc.api.app:app --host 0.0.0.0 --port 8000
```

The interactive API docs are available at `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc` (ReDoc).

### Quick Example

```bash
# 1. Load a voice model
curl -X POST http://localhost:8000/api/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_path": "/path/to/model.pth"}'
# Response: {"model_id": "abc123def456", "model_path": "...", "version": "v2", ...}

# 2. Convert audio (upload file)
curl -X POST http://localhost:8000/api/v1/convert \
  -F "audio=@input.wav" \
  -F "model_id=abc123def456" \
  -F "pitch=12" \
  -F "f0_method=rmvpe" \
  -o output.wav

# 3. Convert audio (server file path)
curl -X POST http://localhost:8000/api/v1/convert/file \
  -H "Content-Type: application/json" \
  -d '{"model_id": "abc123def456", "input_path": "/path/to/input.wav", "pitch": 12}'

# 4. Unload model
curl -X DELETE http://localhost:8000/api/v1/models/abc123def456
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Health check & system status |
| `GET` | `/api/v1/config` | Current server configuration |
| `GET` | `/api/v1/methods` | List available F0 methods |
| `GET` | `/api/v1/embedders` | List available embedder models |
| `POST` | `/api/v1/models/load` | Load a voice model into memory |
| `GET` | `/api/v1/models` | List all loaded models |
| `GET` | `/api/v1/models/{model_id}` | Get info about a loaded model |
| `DELETE` | `/api/v1/models/{model_id}` | Unload a model from memory |
| `POST` | `/api/v1/convert` | Convert audio (file upload, returns audio stream) |
| `POST` | `/api/v1/convert/file` | Convert audio (server paths, returns output path) |

For full request/response schemas, see the interactive docs at `/docs`.

## Models

**Using CLI:**
- Specify the model path with the `-m` option
- Specify the index path with the `-idx` option (recommended for better voice quality)

**Model Download:**
Predictor and embedder models are automatically downloaded from HuggingFace. When using the Python API, models are preloaded at `Config()` initialization for faster first inference. The CLI and API server also preload models at startup.

**Note:** Pre-trained RVC models (.pth files) can be downloaded from various sources. Ensure you have the right to use any model before converting audio with it.

## Recent Bug Fixes

The following bugs have been identified and fixed in this repository:

### Critical Fixes
- **HiFi-GAN Generator**: `forward()` method was incorrectly nested inside `__init__()`, making the vocoder completely non-functional
- **ONNX Export**: `SynthesizerONNX` class didn't exist, causing `ImportError` on import
- **FCPE Model**: `SelfAttention` was missing `use_norm` parameter, causing `TypeError` during model creation
- **F0 Method Lookup**: `compute_f0()` was splitting method names on `-`, causing all CREPE and legacy methods to silently fall back to PM
- **Noisereduce Import**: Wrong import path `rvc.lib.tools.noisereduce` (should be `rvc.tools.noisereduce`) — noise reduction was broken
- **DJCM OpenCL**: Wrong import `from main.library.backends.utils import STFT` — DJCM on OpenCL devices was broken
- **PyWorld DLL**: Was attempting to open the `assets` directory as a file instead of `assets/models/world.bin`
- **Memory Cleanup**: `VoiceConverter.cleanup()` double-deleted `net_g` causing `AttributeError`
- **DJCM Window Length**: `WINDOW_LENGTH` reassignment inside `if svs:` block caused `UnboundLocalError` when `svs=False`

### Significant Fixes
- **Autotune Strength**: `f0_autotune_strength` defaulted to `False` instead of `1.0`, making autotune ineffective by default
- **FCPE Gaussian Blur**: Operator precedence bug caused `*` to bind before `&`, producing boolean tensors instead of float probabilities
- **FCPE Decoder**: Both "argmax" and "local_argmax" decoder options mapped to the same decoder function
- **GELU Activation**: `gelu_accurate()` returned `None` on all calls after the first
- **Fairseq Assertion**: `assert src_len, key_bsz == value.shape[:2]` was parsed as a tuple assertion (always True)
- **Cross-Attention**: `SelfAttention.forward()` left `out` undefined when `cross_attend=True`

### Packaging & Documentation Fixes (v0.1.1)
- **Package imports broken**: `rvc/__init__.py` used `from infer.cli import ...` instead of `from rvc.infer.cli import ...` — `import rvc` raised `ModuleNotFoundError`
- **`run_inference_script` alias missing**: README/DOCUMENTATION/Colab notebook all reference `run_inference_script`, but only `infer_main` was defined — added backward-compatible alias
- **`rvc --version` crashed**: imported `__version__` from the wrong module (`rvc.infer` instead of `rvc`)
- **Missing `__init__.py` files**: `rvc/infer/` and `rvc/lib/` had only stale `.pyc` files; added real `__init__.py`
- **Version / author mismatch**: `rvc.__version__ = "1.0.0"` and `__author__ = "BF667"` did not match `pyproject.toml` (`0.1.0` / `uziproj`)
- **`clean_strength` default divergence**: CLI default was `0.7`, but `VoiceConverter.convert_audio` defaulted to `0.5`
- **Missing `tqdm` dependency**: used in `utils.py` and `gdown.py` but never declared
- **Missing API deps in `requirements.txt`**: `fastapi`, `uvicorn`, `python-multipart` were missing
- **Unused deps**: `tkinter-embed` and `customtkinter` declared but never imported — removed
- **Wrong GitHub URLs**: `pyproject.toml`, README, DOCUMENTATION, Colab notebook, and LICENSE all pointed to `SawitProject/rvc` instead of `uziproj/rvc`
- **Hop-length tip wrong**: docs claimed default was `128`; actual default is `64`
- **No top-level exports**: users had to write `from rvc.infer.infer import run_inference_script` / `from rvc.lib.config import Config` etc. — added re-exports so `from rvc import *` and `from rvc import Config, run_inference_script` both work

For the complete list, see [DOCUMENTATION.md](DOCUMENTATION.md).

## Disclaimer
- The RVC project is developed for research, educational, and personal entertainment purposes. I do not encourage, nor do I take any responsibility for, any misuse of voice conversion technology for fraudulent purposes, identity impersonation, or violations of privacy or copyright belonging to any individual or organization.

- Users are solely responsible for how they use this software and must comply with the laws and regulations of the country in which they reside or operate.

- The use of voices of celebrities, real people, or public figures must be authorized or ensured not to violate any applicable laws, ethical standards, or the rights of the individuals involved.

- The author of this project holds no legal liability for any consequences arising from the use of this software.

## Terms of Use
- You must ensure that any audio content you upload and convert through this project does not infringe upon the intellectual property rights of any third party.

- This project must not be used for any illegal activity, including but not limited to fraud, harassment, or causing harm to others.

- You are fully responsible for any damages resulting from improper use of the product.

- I am not liable for any direct or indirect damages arising from the use of this project.

## Documentation

For detailed documentation, including API references, troubleshooting guides, and advanced usage, see [DOCUMENTATION.md](DOCUMENTATION.md).

## Project-based construction

- **Algorithm: [Vietnamese RVC](https://github.com/PhamHuynhAnh16/Vietnamese-RVC)**
