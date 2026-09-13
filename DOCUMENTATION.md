# RVC (Retrieval-based Voice Conversion) Documentation

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Requirements](#requirements)
- [Python and NumPy Compatibility](#python-and-numpy-compatibility)
- [Features](#features)
- [Command-Line Interface](#command-line-interface)
- [Python API](#python-api)
- [Google Colab](#google-colab)
- [REST API](#rest-api)
- [Architecture](#architecture)
- [Bug Fix Changelog](#bug-fix-changelog)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

RVC (Retrieval-based Voice Conversion) is a simple, high-quality voice conversion tool focused on simplicity and ease of use. It enables high-quality voice conversion using machine learning models with various pitch extraction methods and advanced features.

## Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12
- FFmpeg installed on your system

### Installation Methods

#### Option 1: Install from Git (Recommended)
```bash
pip install git+https://github.com/uziproj/rvc.git
```

#### Option 2: Install from Source (Development)
```bash
git clone https://github.com/uziproj/rvc.git
cd rvc
pip install -e .
```

## Requirements

### System Requirements
- Operating System: Windows, macOS, or Linux
- Memory: At least 4GB RAM (8GB+ recommended for GPU processing)
- Storage: At least 500MB free space for installation
- Audio: Microphone for input, speakers/headphones for output

### Software Requirements
- Python 3.10-3.12
- FFmpeg (for audio processing)
- Compatible GPU (optional, for accelerated processing)

### Python Dependencies
- torch >= 2.3.1
- torchvision >= 0.18.1
- torchaudio >= 2.3.1
- numpy >= 1.25.2 (compatible with 2.0+)
- librosa >= 0.10.2
- soundfile >= 0.13.0
- faiss-cpu >= 1.7.3
- einops >= 0.8.0
- praat-parselmouth
- scipy >= 1.15.0
- omegaconf >= 2.0.6
- transformers >= 4.49.0
- And more (see pyproject.toml for complete list)

## Python and NumPy Compatibility

### Supported Python Versions
This project officially supports Python versions 3.10, 3.11, and 3.12. The code has been tested and verified to work correctly across all these versions.

### NumPy Compatibility
The project has been updated to be compatible with newer NumPy versions (2.0+), resolving compatibility issues that previously existed with Python 3.11+ installations. Key improvements include:

- Replaced deprecated `np.rint()` with `np.round()` for better compatibility
- Added fallback import mechanisms for NumPy functions that may vary between versions
- Maintained backward compatibility with older NumPy versions where possible

### Known Issues
- NumPy 2.0 introduced some API changes that required code adjustments
- Some older NumPy functions have been deprecated and replaced with newer equivalents

## Features

### Pitch Extraction Methods

RVC supports over 20 different pitch extraction methods spanning classical DSP algorithms and modern deep-learning models. Additionally, hybrid methods allow combining multiple extractors (e.g., `hybrid[rmvpe+fcpe]`) for more robust pitch detection.

#### Classical / DSP-Based Methods

| Method | Paper / Source | Description |
|--------|----------------|-------------|
| `pm` | Boersma, "Accurate short-term analysis of the fundamental frequency and the harmonics-to-noise ratio of a sampled sound", *Proc. IFA*, 1993 ([PDF](https://www.fon.hum.uva.nl/paul/papers/Proceedings_1993.pdf)) | Autocorrelation-based pitch detection via the Praat algorithm, accessed through the [Parselmouth](https://github.com/YannickJadoul/Parselmouth) Python binding. The fastest method available; good for quick testing but less accurate on noisy or polyphonic audio. |
| `dio` | Morise & Kawahara, "Fast and reliable F0 estimation method based on the period extraction of vocal fold vibration", *IEICE*, 2010 | The DIO (Distributed Inline Operation) algorithm from the [WORLD](https://github.com/mmorise/World) vocoder. Estimates F0 by detecting periodicity via temporal envelope analysis. Fast but less accurate; intended as a lightweight front-end for real-time applications. Followed by StoneMask refinement in this implementation. |
| `harvest` | Morise, "Harvest: A high-performance fundamental frequency estimator from speech signals", *Interspeech*, 2017 ([PDF](https://www.isca-archive.org/interspeech_2017/morise17b_interspeech.pdf)) | The Harvest algorithm from the WORLD vocoder. An improved F0 estimator over DIO that combines fundamental component extraction with instantaneous frequency analysis. Significantly more accurate than DIO, especially on speech with dynamic F0 contours. Also followed by StoneMask refinement. |
| `yin` | de Cheveigné & Kawahara, "YIN, a fundamental frequency estimator for speech and music", *JASA*, 2002 ([PDF](http://audition.ens.fr/adc/pdf/2002_JASA_YIN.pdf)) | The YIN algorithm uses a modified autocorrelation with a cumulative mean normalised difference function and an absolute threshold to avoid octave errors. Implemented via [librosa](https://librosa.org/). Deterministic — produces a single F0 estimate per frame without voicing uncertainty. |
| `pyin` | Mauch & Dixon, "pYIN: A fundamental frequency estimator using probabilistic threshold distributions", *ICASSP*, 2014 ([PDF](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/6040/MAUCHpYINFundamental2014Accepted.pdf)) | A probabilistic extension of YIN. Instead of a single threshold, pYIN samples multiple thresholds to produce a set of pitch candidates per frame, then uses a hidden Markov model (HMM) with Viterbi decoding to select the most likely pitch trajectory. More robust than YIN in noisy conditions and better at handling voicing decisions. Implemented via librosa. |
| `swipe` | Camacho & Harris, "A sawtooth waveform inspired pitch estimator for speech and music", *JASA*, 2008 ([DOI](https://pubs.aip.org/asa/jasa/article/124/3/1638/676279)) | SWIPE (Sawtooth Waveform Inspired Pitch Estimator) estimates pitch by finding the fundamental frequency of the sawtooth waveform whose spectrum best matches the input signal's spectrum. Operates in the frequency domain and is particularly robust for musical signals. Followed by StoneMask refinement in this implementation. |

#### Deep-Learning Methods

| Method | Paper / Source | Description |
|--------|----------------|-------------|
| `rmvpe` | Wei, Cao, Dan & Chen, "RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music", *Interspeech*, 2023 ([arXiv](https://arxiv.org/abs/2306.15412)) | Uses a deep U-Net architecture to extract hidden features from the spectrogram and predict vocal pitches even in the presence of musical accompaniment. The recommended default method in RVC — achieves the best balance of accuracy and robustness for voice conversion. |
| `rmvpe-legacy` | Same as `rmvpe` | Uses the same RMVPE model but applies additional StoneMask-based pitch refinement (`infer_from_audio_with_pitch`) that clips F0 values to the configured `f0_min`/`f0_max` range. Can reduce pitch outliers at the cost of slightly over-smoothing the F0 contour. |
| `fcpe` | CNChTu, "FCPE: A Fast Context-based Pitch Estimation Model", *arXiv*, 2025 ([arXiv](https://arxiv.org/abs/2509.15140), [GitHub](https://github.com/CNChTu/FCPE)) | Employs a Lynx-Net architecture with depth-wise separable convolutions and full-context attention for fast, accurate pitch estimation. Uses a lower voicing threshold (0.006) than the legacy variant, making it more sensitive to voiced regions. |
| `fcpe-legacy` | Same as `fcpe` | The original FCPE implementation with a higher voicing threshold (0.03), which is more conservative in detecting voiced frames. Better for clean speech but may miss weaker voiced segments. |
| `crepe-{tiny\|small\|medium\|large\|full}` | Kim, Cremer, Shah, Tan, Toshniwal, Staum & Ellis, "CREPE: A Convolutional Representation for Pitch Estimation", *ISMIR*, 2018 ([arXiv](https://arxiv.org/abs/1802.06182), [GitHub](https://github.com/marl/crepe)) | A deep CNN operating directly on the time-domain waveform. Outputs a 360-bin pitch distribution per frame, decoded via Viterbi. Five model sizes are available: `tiny` (fastest/least accurate) through `full` (slowest/most accurate). This implementation applies mean filtering to the pitch and median filtering to periodicity, zeroing out frames with periodicity below 0.1. |
| `mangio-crepe-{tiny\|small\|medium\|large\|full}` | Same model as `crepe-*`, from the [Mangio-RVC-Fork](https://github.com/Mangio621/Mangio-RVC-Fork) | Uses the same CREPE CNN model weights but disables periodicity filtering — returns the raw Viterbi-decoded pitch without the mean/median smoothing and voiced/unvoiced gating applied in the standard `crepe-*` mode. Produces smoother, more continuous pitch contours that some users prefer for singing voice conversion. |
| `djcm` | Wei, Cao, Xu & Chen, "DJCM: A Deep Joint Cascade Model for Singing Voice Separation and Vocal Pitch Estimation", *ICASSP*, 2024 ([arXiv](https://arxiv.org/abs/2401.03856)) | A joint cascade model that simultaneously performs singing voice separation and vocal pitch estimation. The SVS module extracts clean vocals from polyphonic audio, and the VPE module estimates pitch from the separated vocals. Particularly effective for music with heavy accompaniment. |

#### Hybrid Methods

| Syntax | Description |
|--------|-------------|
| `hybrid[method1+method2]` | Combines multiple F0 methods using a weighted geometric mean. Example: `hybrid[rmvpe+fcpe]` runs both RMVPE and FCPE and merges their F0 contours. The weighting uses a quadratic function centered at `alpha=0.5` (equal weight). Useful for getting the best of multiple methods — e.g., combining RMVPE's robustness with FCPE's speed. |

#### Method Selection Guide

- **Quick testing / prototyping**: `pm` or `dio` — fast but lower quality
- **General voice conversion**: `rmvpe` — best overall accuracy, recommended default
- **Singing with accompaniment**: `djcm` — joint separation + pitch estimation
- **Fast high-quality inference**: `fcpe` — Lynx-Net architecture, faster than RMVPE
- **Maximum accuracy**: `crepe-full` or `mangio-crepe-full` — largest CREPE model, slowest
- **Smooth singing voice**: `mangio-crepe-large` — no periodicity gating, smoother contours
- **Robustness via ensemble**: `hybrid[rmvpe+fcpe]` — combines two deep-learning methods

### Advanced Features
- **Formant Shifting**: Modify vocal tract characteristics independently of pitch using STFT-based pitch shifting
- **Noise Reduction**: Clean up audio output using spectral gating noise reduction
- **Autotune**: Automatic pitch correction with adjustable strength (0.0-1.0, default 1.0)
- **Batch Processing**: Process multiple audio files at once via directory input
- **Model Indexing**: Use .index files with FAISS for better voice quality preservation through feature retrieval
- **Proposal Pitch**: Automatically calculate optimal pitch shift based on source audio characteristics
- **Hybrid F0**: Combine multiple pitch extraction methods for improved robustness

### Embedder Models
- **ContentVec** (default): General-purpose voice embedding model
- **HuBERT Base**: Multilingual support including Japanese, Korean, Chinese, and Portuguese variants
- **SPIN**: Speaker Independent Network

### Vocoder Options
- **Default**: HiFi-GAN NSF (Neural Source Filter) for f0 models
- **RefineGAN**: Alternative high-quality vocoder
- **MRF-HiFi-GAN**: Multi-Receptive Field HiFi-GAN
- **HiFi-GAN**: Standard vocoder for non-f0 models

## Command-Line Interface

### Basic Usage
```bash
rvc -i input.wav -o output.wav -m model.pth
```

### Options

#### Basic Options
- `-i, --input`: Path to input audio file or directory (required)
- `-o, --output`: Path to output audio file (default: ./output.wav)
- `-m, --model`: Path to .pth model file (required)

#### Common Options
- `-p, --pitch`: Pitch shift in semitones (default: 0)
- `-f0, --f0_method`: F0 prediction method (default: rmvpe)
- `-idx, --index`: Path to .index file
- `-ir, --index_rate`: Index rate for feature retrieval (default: 0.5)
- `-em, --embedder`: Embedder model name (default: contentvec_base)

#### Advanced Options
- `-split, --split_audio`: Split audio into chunks for processing
- `-clean, --clean_audio`: Apply noise reduction to output
- `-cs, --clean_strength`: Noise reduction strength 0.0-1.0 (default: 0.7)
- `-fa, --f0_autotune`: Enable F0 autotune
- `-fas, --f0_autotune_strength`: Autotune strength 0.0-1.0 (default: 1.0)
- `-fs, --formant_shifting`: Enable formant shifting
- `-fq, --formant_qfrency`: Formant frequency parameter (default: 0.8)
- `-ft, --formant_timbre`: Formant timbre parameter (default: 0.8)
- `-pp, --proposal_pitch`: Enable proposal pitch
- `-ppt, --proposal_pitch_threshold`: Proposal pitch threshold (default: 255.0)
- `-fr, --filter_radius`: Filter radius for pitch extraction (default: 3)
- `-hl, --hop_length`: Hop length for pitch extraction (default: 64)
- `-rs, --resample_sr`: Resample output audio sample rate (default: 0, no resampling)
- `-fmt, --format`: Output format: wav, flac, mp3, ogg (default: wav)

### Example Usage
```bash
# Simple conversion
rvc -i input.wav -o output.wav -m model.pth -p 12

# Batch conversion from directory
rvc -i ./audio_folder -m model.pth -p 12 -f0 rmvpe

# With index file and autotune
rvc -i input.wav -m model.pth -idx model.index -ir 0.75 -fa

# With formant shifting
rvc -i input.wav -o output.wav -m model.pth -fs -fq 0.9 -ft 0.7

# Advanced processing with multiple options
rvc -i input.wav -o output.wav -m model.pth -p -5 -f0 rmvpe -idx model.index -ir 0.8 -fa -fas 0.8 -clean

# Using CREPE large model for highest quality
rvc -i input.wav -o output.wav -m model.pth -f0 crepe-large

# Hybrid method combining RMVPE and FCPE
rvc -i input.wav -o output.wav -m model.pth -f0 "hybrid[rmvpe+fcpe]"
```

## Python API

### Basic Inference
```python
from rvc.infer.infer import run_inference_script
from rvc.lib.config import Config

# Hubert and RMVPE models are preloaded at Config initialization
# Pass embedder_model and f0_method to specify which models to load eagerly
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

### Using VoiceConverter Directly
```python
from rvc.infer.cli import VoiceConverter
from rvc.lib.config import Config

# Models preload at startup — hubert (embedder) and rmvpe (predictor)
config = Config(embedder_model="contentvec_base", f0_method="rmvpe")
converter = VoiceConverter(config, model_path="model.pth", sid=0)

converter.convert_audio(
    audio_input_path="input.wav",
    audio_output_path="output.wav",
    index_path="model.index",
    embedder_model="contentvec_base",
    pitch=12,
    f0_method="rmvpe",
    index_rate=0.5,
    volume_envelope=1.0,
    protect=0.5,
    hop_length=64,
    filter_radius=3,
    export_format="wav",
)
```

### Pitch Extraction
```python
from rvc.lib.predictor.generator import Generator
from rvc.lib.config import Config

# Preload RMVPE via Config, then pass config to Generator
config = Config(embedder_model="contentvec_base", f0_method="rmvpe")
gen = Generator(sample_rate=16000, hop_length=160, f0_min=50, f0_max=1100, is_half=False, device="cpu", config=config)
f0_mel, f0_hz = gen.calculator(
    f0_method="rmvpe",
    x=audio_data,
    f0_up_key=0,
    p_len=None,
    filter_radius=3,
    f0_autotune=True,
    f0_autotune_strength=0.8,
)
```

### Autotune
```python
from rvc.utils import Autotune

ref_freqs = [49.00, 51.91, 55.00, 58.27, 61.74, 65.41, 69.30, 73.42, 77.78, 82.41,
             87.31, 92.50, 98.00, 103.83, 110.00, 116.54, 123.47, 130.81, 138.59, 146.83]
autotuner = Autotune(ref_freqs)
adjusted_f0 = autotuner.autotune_f0(original_f0, f0_autotune_strength=0.8)
```

## Google Colab

A ready-to-run Colab notebook is available for quick testing without local installation:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uziproj/rvc/blob/main/colab/rvc_demo.ipynb)

The notebook includes:
- Automatic installation of RVC and dependencies
- Model download from HuggingFace
- Simple voice conversion demo
- Batch processing example

## REST API

RVC includes a built-in REST API server powered by FastAPI, enabling voice conversion over HTTP. This is ideal for integrating RVC into web applications, microservices, or any system that communicates via HTTP.

### Starting the Server

```bash
# Using the CLI entry point
rvc-api --host 0.0.0.0 --port 8000

# Or with uvicorn directly
uvicorn rvc.api.app:app --host 0.0.0.0 --port 8000 --workers 1

# Development mode with auto-reload
rvc-api --host 0.0.0.0 --port 8000 --reload
```

Once running, interactive API documentation is available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Health check and system status |
| `GET` | `/api/v1/config` | Current server configuration (device, precision, etc.) |
| `GET` | `/api/v1/methods` | List all available F0 extraction methods |
| `GET` | `/api/v1/embedders` | List all available embedder models |
| `POST` | `/api/v1/models/load` | Load a .pth voice model into memory |
| `GET` | `/api/v1/models` | List all currently loaded models |
| `GET` | `/api/v1/models/{model_id}` | Get detailed info about a loaded model |
| `DELETE` | `/api/v1/models/{model_id}` | Unload a model and free GPU memory |
| `POST` | `/api/v1/convert` | Convert audio via file upload (returns audio stream) |
| `POST` | `/api/v1/convert/file` | Convert audio via server file paths (returns output path) |

### Usage Workflow

The typical workflow involves three steps: load a model, convert audio, then optionally unload the model.

#### Step 1: Load a Voice Model

```bash
curl -X POST http://localhost:8000/api/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_path": "/path/to/voice_model.pth", "sid": 0}'
```

Response:
```json
{
  "model_id": "a1b2c3d4e5f6",
  "model_path": "/path/to/voice_model.pth",
  "version": "v2",
  "vocoder": "Default",
  "use_f0": true,
  "n_spk": 1,
  "target_sr": 40000
}
```

The `model_id` is a deterministic hash of the model file path. Loading the same file again returns the existing model's info without re-loading.

#### Step 2a: Convert Audio (File Upload)

Upload an audio file and receive the converted result as a streaming audio response:

```bash
curl -X POST http://localhost:8000/api/v1/convert \
  -F "audio=@my_voice.wav" \
  -F "model_id=a1b2c3d4e5f6" \
  -F "pitch=12" \
  -F "f0_method=rmvpe" \
  -F "embedder_model=contentvec_base" \
  -F "index_rate=0.5" \
  -F "output_format=wav" \
  -o converted.wav
```

All conversion parameters (pitch, f0_method, clean_audio, formant_shifting, etc.) can be passed as form fields. See the interactive docs for the full parameter list.

#### Step 2b: Convert Audio (Server File Path)

If the input audio is already on the server, use the JSON-based endpoint:

```bash
curl -X POST http://localhost:8000/api/v1/convert/file \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "a1b2c3d4e5f6",
    "input_path": "/data/audio/input.wav",
    "output_format": "wav",
    "pitch": 12,
    "f0_method": "rmvpe",
    "embedder_model": "contentvec_base",
    "index_path": "/data/models/voice.index",
    "index_rate": 0.75,
    "clean_audio": true,
    "clean_strength": 0.7,
    "f0_autotune": true,
    "f0_autotune_strength": 0.8
  }'
```

Response:
```json
{
  "success": true,
  "output_path": "/data/audio/input_converted.wav",
  "message": "Conversion complete. Output saved to /data/audio/input_converted.wav"
}
```

#### Step 3: Unload the Model

Free GPU memory when done:

```bash
curl -X DELETE http://localhost:8000/api/v1/models/a1b2c3d4e5f6
```

### Python Client Example

```python
import requests

API = "http://localhost:8000/api/v1"

# Load model
resp = requests.post(f"{API}/models/load", json={
    "model_path": "/path/to/model.pth"
})
model_id = resp.json()["model_id"]
print(f"Loaded model: {model_id}")

# Convert audio (upload)
with open("input.wav", "rb") as f:
    resp = requests.post(f"{API}/convert", files={
        "audio": f
    }, data={
        "model_id": model_id,
        "pitch": "12",
        "f0_method": "rmvpe",
        "output_format": "wav",
    })

with open("output.wav", "wb") as f:
    f.write(resp.content)
print("Conversion complete!")

# Unload model
requests.delete(f"{API}/models/{model_id}")
```

### Model Management

The API maintains an in-memory model registry. Models remain loaded between requests, so you only need to load them once. This is efficient for serving multiple conversion requests with the same model. Key behaviors include:

- **Deterministic IDs**: The `model_id` is derived from the model file path, so re-loading the same file returns the existing model
- **Eager model loading**: Hubert (embedder) and RMVPE (predictor) models are preloaded at server startup inside `Config()`, eliminating lazy-loading latency on the first conversion request
- **Automatic predictor download**: F0 predictor models are automatically downloaded from HuggingFace if not found locally
- **GPU memory management**: Use `DELETE /api/v1/models/{model_id}` to explicitly free GPU memory when a model is no longer needed

### Concurrency and Performance

- **Async with thread pool**: The API uses `asyncio.to_thread()` to run blocking inference operations in a thread pool, keeping the event loop responsive
- **Single-worker recommended**: For GPU inference, use `--workers 1` (default) since GPU operations are typically serialized anyway
- **Multiple models**: You can load multiple voice models simultaneously; each is cached independently in the registry
- **Streaming responses**: The `/convert` endpoint streams the output audio directly, avoiding large in-memory buffers

## Architecture

### Project Structure
```
rvc/
  __init__.py
  utils.py              # Utility functions (audio loading, GPU cache, downloads)
  var.py                # Global variables (f0 method list)
  api/                  # REST API server
    __init__.py
    app.py              # FastAPI application and endpoints
  lib/
    algorithm/          # Core ML algorithm components
      commons.py        # Shared utilities, weight init
      attentions.py     # Attention mechanisms
      encoders.py       # Text and posterior encoders
      residuals.py      # Residual coupling blocks
      synthesizers.py   # Main Synthesizer model
      normalization.py  # Normalization layers
      modules.py        # Building block modules
    backend/            # Hardware backend support
      opencl.py         # AMD OpenCL support
      rms.py            # RMS energy extraction
      stftpitchshift.py # STFT pitch shifting
      zulda.py          # Additional backend
    embedders/          # Voice embedding models
      fairseq.py        # Fairseq/HuBERT embedder
    generators/         # Vocoder generators
      hifigan.py        # HiFi-GAN vocoder
      nsf_hifigan.py    # NSF HiFi-GAN vocoder
      refinegan.py      # RefineGAN vocoder
      mrf_hifigan.py    # MRF HiFi-GAN vocoder
    onnx/               # ONNX export support
      onnx_export.py    # Model export to ONNX
      wrapper.py        # ONNX inference wrapper
    predictor/          # Pitch estimation algorithms
      generator.py      # F0 computation dispatcher
      rmvpe.py          # RMVPE pitch estimator
      torchfcpe.py      # FCPE pitch estimator
      torchcrepe.py     # CREPE pitch estimator
      djcm.py           # DJCM pitch estimator
      pyworld.py        # PyWorld (DIO/Harvest) estimator
      swipe.py          # SWIPE pitch estimator
    config.py           # Configuration (device, paths, eager model loading)
  infer/                # Inference pipeline
    cli.py              # Command-line interface
    infer.py            # Python API entry point
    pipeline.py         # Voice conversion pipeline
  tools/                # Utility tools
    cut.py              # Audio splitting/restoration
    gdown.py            # Google Drive downloader
    noisereduce.py      # Noise reduction
  tts/                  # Text-to-speech integration
    infer.py            # TTS inference
```

### Inference Pipeline Flow
1. **Configuration**: `Config()` initializes device settings and preloads Hubert + RMVPE models
2. **Audio Loading**: Load and resample input audio to 16kHz
3. **Embedding Extraction**: Extract voice features using preloaded HuBERT/ContentVec
4. **Feature Retrieval** (optional): Enhance features using FAISS index
5. **Pitch Estimation**: Extract F0 contour using preloaded RMVPE (or selected method)
6. **Autotune** (optional): Snap pitch to nearest musical notes
7. **Voice Conversion**: Run synthesizer model with features + pitch
8. **Post-processing**: Volume envelope matching, noise reduction, resampling

### Eager Model Loading

By default, `Config()` preloads two heavy models at initialization time so that the first inference call is fast:

- **Hubert/ContentVec** (embedder): Extracts voice features from audio. Controlled via the `embedder_model` parameter (default: `"contentvec_base"`).
- **RMVPE** (pitch predictor): Estimates fundamental frequency (F0). Controlled via the `f0_method` parameter (default: `"rmvpe"").

```python
from rvc.lib.config import Config

# Default: loads contentvec_base + rmvpe
config = Config()

# Custom embedder and predictor
config = Config(embedder_model="hubert_base", f0_method="crepe-full")
```

If a model file is not found locally, it is automatically downloaded from HuggingFace. If loading fails for any reason, the system gracefully falls back to lazy loading (loading on first use).

## Bug Fix Changelog

### Critical Fixes

#### BUG 1: HiFi-GAN Generator `forward()` inaccessible
- **File**: `rvc/lib/generators/hifigan.py`
- **Description**: The `forward()` method was defined as a nested function inside `__init__()` instead of as a class method. PyTorch calls `model.forward()` when invoking `model(x)`, which would fail with `NotImplementedError` since no class-level `forward` existed. This made the entire HiFi-GAN vocoder non-functional.
- **Fix**: Dedented the `forward()` method so it is a proper class method at the correct indentation level.

#### BUG 2: `SynthesizerONNX` class doesn't exist
- **File**: `rvc/lib/onnx/onnx_export.py`
- **Description**: The ONNX export module imported `SynthesizerONNX` from `synthesizers.py`, but only `Synthesizer` was defined there. This caused an `ImportError` at module load time, breaking all ONNX export functionality.
- **Fix**: Changed the import to `from rvc.lib.algorithm.synthesizers import Synthesizer as SynthesizerONNX`, since the same `Synthesizer` class can serve both purposes.

#### BUG 3: `SelfAttention.__init__` missing `use_norm` parameter
- **File**: `rvc/lib/predictor/torchfcpe.py`
- **Description**: `CFNEncoderLayer` called `SelfAttention(dim=dim_model, heads=num_heads, causal=False, use_norm=use_norm, ...)` but `SelfAttention.__init__` did not accept `use_norm`. This raised `TypeError` whenever the FCPE model was instantiated.
- **Fix**: Added `use_norm=None` parameter to `SelfAttention.__init__` signature.

#### BUG 4: `FastAttention.causal_linear_fn` doesn't exist
- **File**: `rvc/lib/predictor/torchfcpe.py`
- **Description**: When `self.causal=True`, `forward()` referenced `self.causal_linear_fn`, which was never defined. This would raise `AttributeError` if causal attention was ever used.
- **Fix**: Implemented `_causal_linear_fn` as a static method using cumulative sum-based linear attention.

#### BUG 5: `compute_f0()` method name splitting breaks CREPE and legacy methods
- **File**: `rvc/lib/predictor/generator.py`
- **Description**: `base_method = f0_method.split('-')[0]` was used for dict lookup, causing: `"mangio-crepe-tiny"` to resolve to `"mangio"` (not found, falls back to PM), `"crepe-tiny"` to resolve to `"crepe"` (not found, falls back to PM), and `"rmvpe-legacy"` to resolve to `"rmvpe"` (calling non-legacy function). All CREPE variants and legacy methods were silently broken.
- **Fix**: Removed the `base_method` splitting logic and used the full `f0_method` string for dict lookup.

#### BUG 6: Wrong import path for `noisereduce`
- **File**: `rvc/infer/cli.py`, `rvc/infer/infer.py`
- **Description**: Both files imported `from rvc.lib.tools.noisereduce import reduce_noise`, but `noisereduce.py` lives at `rvc/tools/noisereduce.py`, not `rvc/lib/tools/noisereduce.py`. This caused `ModuleNotFoundError` whenever `clean_audio` was enabled.
- **Fix**: Changed import to `from rvc.tools.noisereduce import reduce_noise`.

#### BUG 7: Wrong import in DJCM for OpenCL devices
- **File**: `rvc/lib/predictor/djcm.py`
- **Description**: `Spectrogram.forward()` imported `from main.library.backends.utils import STFT` for OpenCL devices, but this path does not exist in the project. The correct import should be from the project's own `opencl` backend module.
- **Fix**: Changed to `from rvc.lib.backend.opencl import STFT`.

#### BUG 8: `PYWORLD.__init__` opens a directory instead of a file
- **File**: `rvc/lib/predictor/pyworld.py`
- **Description**: `open(os.path.join("assets"), "rb")` attempted to open the `"assets"` directory as a file for `pickle.load()`, raising `IsADirectoryError` on Linux/macOS or `PermissionError` on Windows when the WORLD DLL was not yet extracted.
- **Fix**: Changed to `open(os.path.join("assets", "models", "world.bin"), "rb")` to correctly reference the pickle file containing the DLL data.

#### BUG 9: `VoiceConverter.cleanup()` double-deletes `self.net_g`
- **File**: `rvc/infer/cli.py`, `rvc/infer/infer.py`
- **Description**: The `cleanup()` method first deleted `self.net_g` inside the `if self.hubert_model is not None` block, then attempted `del self.net_g` again in a separate `if hasattr(self, 'net_g')` block. Since `net_g` was already deleted (along with other attributes) in the first block, `hasattr(self, 'net_g')` could still be True in edge cases, leading to `AttributeError`.
- **Fix**: Removed the redundant `if hasattr(self, 'net_g'): del self.net_g` block, keeping only the proper cleanup within the `hubert_model` check.

#### BUG 10: DJCM `WINDOW_LENGTH` local variable causes `UnboundLocalError`
- **File**: `rvc/lib/predictor/djcm.py`
- **Description**: Line `WINDOW_LENGTH = 2048` inside an `if svs:` block shadowed the module-level constant. Python treats any variable assigned anywhere in a function as local throughout that function, so when `svs=False`, accessing `WINDOW_LENGTH` raised `UnboundLocalError`.
- **Fix**: Used a local variable `window_length = 2048 if svs else WINDOW_LENGTH` instead of reassigning the module constant.

### Significant Fixes

#### BUG 11: `f0_autotune_strength` defaults to `False`
- **File**: `rvc/infer/pipeline.py`
- **Description**: `f0_autotune_strength=False` meant `False` evaluates to `0` in numeric context, so `autotune_f0()` returned the original f0 unmodified (since `if f0_autotune_strength <= 0: return f0`). Autotune had no effect with the default parameter.
- **Fix**: Changed default to `f0_autotune_strength=1.0`.

#### BUG 12: `gaussian_blurred_cent` operator precedence bug
- **File**: `rvc/lib/predictor/torchfcpe.py`
- **Description**: The expression `torch.exp(...) * (cents > 0.1) & (cents < ...)` evaluated as `(torch.exp(...) * (cents > 0.1)) & (cents < ...)` because `*` has higher precedence than `&`. The intended behavior was `torch.exp(...) * ((cents > 0.1) & (cents < ...))`, producing a float probability tensor for `F.binary_cross_entropy`.
- **Fix**: Added explicit parentheses: `* ((cents > 0.1) & (cents < ...)).float()`.

#### BUG 13: Both "argmax" and "local_argmax" decoders map to the same function
- **File**: `rvc/lib/predictor/torchfcpe.py`
- **Description**: In `CFNaiveMelPE.infer()`, when `decoder == "argmax"`, the code assigned `cents = self.latent2cents_local_decoder` instead of `self.latent2cents_decoder`, making the argmax decoder option non-functional.
- **Fix**: Changed to `if decoder == "argmax": cents = self.latent2cents_decoder`.

#### BUG 14: `gelu_accurate` returns `None` on subsequent calls
- **File**: `rvc/lib/embedders/fairseq.py`
- **Description**: The `return` statement was inside the `if not hasattr(gelu_accurate, "_a")` block. On the first call, it set `_a` and returned correctly. On subsequent calls, it skipped the `if` block and had no `return` statement, returning `None`.
- **Fix**: Moved the `return` statement outside the `if` block so it executes on every call.

#### BUG 15: `assert src_len, key_bsz == value.shape[:2]` always True
- **File**: `rvc/lib/embedders/fairseq.py`
- **Description**: Python parsed `assert src_len, key_bsz == value.shape[:2]` as a tuple assertion: `assert (src_len,)` with the comparison as the error message. Since `src_len` (a dimension) is always truthy, the assertion was always True regardless of whether the shapes matched.
- **Fix**: Changed to `assert src_len == value.shape[0] and key_bsz == value.shape[1]`.

#### BUG 16: `SelfAttention.forward` doesn't handle `cross_attend=True`
- **File**: `rvc/lib/predictor/torchfcpe.py`
- **Description**: When `cross_attend` was True (i.e., context was provided), the code just `pass`ed, leaving `out` undefined. Then `attn_outs.append(out)` raised `UnboundLocalError`.
- **Fix**: Changed the `pass` to `out = self.fast_attention(q, k, v)` to properly compute cross-attention output.

### Minor Fixes

#### BUG 17: Duplicate `nn` import in DJCM
- **File**: `rvc/lib/predictor/djcm.py`
- **Fix**: Consolidated `import torch.nn as nn` and `from torch import nn` into a single `from torch import nn`.

#### BUG 18: Duplicate `self.resample_kernel = {}` in RMVPE
- **File**: `rvc/lib/predictor/rmvpe.py`
- **Fix**: Removed the duplicate assignment.

#### BUG 19: `MelModule.__call__` overrides `nn.Module.__call__`
- **File**: `rvc/lib/predictor/torchfcpe.py`
- **Description**: Both `MelModule` and `Wav2MelModule` overrode `__call__` directly instead of implementing `forward`. This bypassed PyTorch's `nn.Module.__call__` mechanism (hooks, `torch.no_grad()` context, etc.).
- **Fix**: Renamed `__call__` to `forward` in both classes.

#### BUG 20: `pytorch_ocl != None` instead of `is not None`
- **File**: `rvc/lib/backend/opencl.py`
- **Fix**: Changed to `pytorch_ocl is not None` per PEP 8.

#### BUG 21: Redundant import fallback in `stftpitchshift.py`
- **File**: `rvc/lib/backend/stftpitchshift.py`
- **Description**: Both `try` and `except` blocks imported `sliding_window_view` from the same path, making the fallback useless.
- **Fix**: Changed the fallback to set `sliding_window_view = None`.

#### BUG 22: `os.path.join("assets")` is misleading
- **File**: `rvc/lib/predictor/pyworld.py`
- **Fix**: Simplified to `"assets"` since `os.path.join` with a single argument just returns that argument.

#### BUG 23: Cookie loading assumes list-of-pairs format
- **File**: `rvc/tools/gdown.py`
- **Description**: `for k, v in cookies:` assumed the JSON file stores cookies as a list of pairs `[[key, val], ...]`. Standard browser cookie exports typically use dict format `{"key": val}`.
- **Fix**: Added `isinstance(cookies, dict)` check to handle both dict and list formats.

### Packaging & Documentation Fixes (v0.1.1)

These bugs were discovered while auditing the package for v0.1.1. They prevented the package from importing at all, made the CLI entry point crash, or caused documentation examples to fail.

#### BUG 24: `rvc/__init__.py` uses non-existent `infer.*` import paths
- **File**: `rvc/__init__.py`
- **Description**: The package init used `from infer.cli import ...` and `from infer.infer import infer_main`. There is no top-level `infer` package — the modules live under `rvc.infer.*`. As a result, `import rvc` raised `ModuleNotFoundError: No module named 'infer'`, breaking every entry point (`rvc`, `rvc-api`) and every documented Python import path.
- **Fix**: Changed imports to `from rvc.infer.cli import ...` and `from rvc.infer.infer import infer_main`.

#### BUG 25: `run_inference_script` referenced in docs but never defined
- **Files**: `README.md`, `DOCUMENTATION.md`, `colab/rvc_demo.ipynb`, `rvc/__init__.py`
- **Description**: All user-facing documentation tells users to call `from rvc.infer.infer import run_inference_script`, but the function defined in `infer.py` is `infer_main`. Every documented Python example would crash with `ImportError`.
- **Fix**: Added a backwards-compatible alias `run_inference_script = infer_main` in `rvc/__init__.py`, `rvc/infer/__init__.py`, and `rvc/infer/infer.py`. Both names now resolve to the same callable; existing scripts that used `infer_main` keep working.

#### BUG 26: `rvc --version` crashes with `ImportError`
- **File**: `rvc/infer/cli.py`
- **Description**: The `--version` handler did `from rvc.infer import __version__`, but `__version__` is defined in `rvc/__init__.py`, not in `rvc.infer`. Running `rvc --version` raised `ImportError`.
- **Fix**: Changed the import to `from rvc import __version__`. Also created a proper `rvc/infer/__init__.py` so the sub-package is a regular package (not an implicit namespace package).

#### BUG 27: Missing `rvc/infer/__init__.py` and `rvc/lib/__init__.py`
- **Files**: `rvc/infer/__init__.py` (missing), `rvc/lib/__init__.py` (missing)
- **Description**: Both sub-directories had only stale `__pycache__/__init__.cpython-312.pyc` files left over from a previous version; the source `__init__.py` had been deleted. This forced Python to use implicit namespace packages, which (a) is unreliable under certain packaging setups and (b) prevents the sub-package from exposing its own `__version__` / public API.
- **Fix**: Added `rvc/infer/__init__.py` (exposing `infer_main`, `run_inference_script`, `VoiceConverter`). Added an empty `rvc/lib/__init__.py`.

#### BUG 28: `__version__` / `__author__` mismatch with `pyproject.toml`
- **File**: `rvc/__init__.py`
- **Description**: `__version__ = "1.0.0"` and `__author__ = "BF667"` did not match `pyproject.toml` (`version = "0.1.0"`, `authors = [{name = "uziproj"}]`). `pip show rvc` and `python -c "import rvc; print(rvc.__version__)"` reported different versions, which confuses dependency resolvers and downstream tooling.
- **Fix**: Aligned both to the values in `pyproject.toml` (`0.1.0` / `uziproj`).

#### BUG 29: `clean_strength` default diverges between CLI and class
- **Files**: `rvc/infer/cli.py` (`VoiceConverter.convert_audio`), `rvc/infer/infer.py` (`VoiceConverter.convert_audio`)
- **Description**: The CLI `argparse` default for `-cs` was `0.7`, but the underlying `VoiceConverter.convert_audio` method defaulted `clean_strength=0.5`. Calling the class directly from Python (as documented in the Python API section) produced noticeably weaker noise reduction than calling via CLI for the same audio.
- **Fix**: Aligned both class defaults to `0.7` to match the CLI and the README/DOCUMENTATION tables.

#### BUG 30: Missing `tqdm` dependency declaration
- **Files**: `pyproject.toml`, `requirements.txt`
- **Description**: `rvc/utils.py` and `rvc/tools/gdown.py` both do `from tqdm import tqdm`, but `tqdm` is not declared in `pyproject.toml` or `requirements.txt`. The import only worked because `tqdm` happens to be a transitive dependency of `gradio` / `transformers`. A clean install with `--no-deps` would crash on the first model download attempt.
- **Fix**: Added `tqdm>=4.65.0` to both `pyproject.toml` (main `dependencies`) and `requirements.txt`.

#### BUG 31: `requirements.txt` missing API dependencies
- **File**: `requirements.txt`
- **Description**: `requirements.txt` did not list `fastapi`, `uvicorn`, or `python-multipart`, even though the `rvc-api` entry point requires all three. Users installing from `requirements.txt` (the conventional `pip install -r requirements.txt` workflow) could not start the API server.
- **Fix**: Added `fastapi>=0.100.0`, `uvicorn>=0.23.0`, `python-multipart>=0.0.6` to `requirements.txt`.

#### BUG 32: Unused dependencies in `requirements.txt`
- **File**: `requirements.txt`
- **Description**: `tkinter-embed` and `customtkinter` were declared but never imported anywhere in the codebase (verified with `grep -r`). They added ~30 MB of unnecessary install weight and pulled in a system-level Tk dependency that breaks headless / server installs.
- **Fix**: Removed both from `requirements.txt`.

#### BUG 33: `pyproject.toml` URLs point to wrong GitHub organization
- **File**: `pyproject.toml`
- **Description**: `[project.urls]` pointed to `github.com/SawitProject/rvc` for Homepage / Repository / Issues, but the canonical repo is `github.com/uziproj/rvc`. `pip show rvc -v` displayed broken links.
- **Fix**: Updated all three URLs to `https://github.com/uziproj/rvc`.

#### BUG 34: Documentation references wrong GitHub URL
- **Files**: `README.md`, `DOCUMENTATION.md`, `colab/rvc_demo.ipynb`, `LICENSE`
- **Description**: Multiple badges, install commands, and "see the documentation" links pointed to `github.com/SawitProject/rvc`. The Colab badge, the troubleshooting install snippet, the Colab notebook footer, and the LICENSE copyright notice were all affected.
- **Fix**: Replaced every `SawitProject/rvc` occurrence with `uziproj/rvc`.

#### BUG 35: DOCUMENTATION.md hop_length tip was misleading
- **File**: `DOCUMENTATION.md`
- **Description**: Performance Tips section said "Use `-hl 128` (default hop length) for best quality; increase for faster processing". This was double-wrong: (1) the actual default is `64`, not `128`; (2) the relationship is inverted — *increasing* `-hl` makes pitch extraction faster and lower quality, not the other way around.
- **Fix**: Rewrote to "Use `-hl 128` for higher-quality pitch extraction (default is `64`; increase to trade speed for accuracy)".

## Troubleshooting

### Common Issues

#### Installation Issues
- **Problem**: Installation fails due to dependency conflicts
- **Solution**: Create a fresh virtual environment and install dependencies separately:
  ```bash
  python -m venv venv
  source venv/bin/activate  # Linux/macOS
  venv\Scripts\activate     # Windows
  pip install git+https://github.com/uziproj/rvc.git
  ```

- **Problem**: PyTorch installation fails
- **Solution**: Install PyTorch separately using the official PyTorch installation guide before installing RVC:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

#### Runtime Issues
- **Problem**: Out of memory errors during inference
- **Solution**: Reduce audio length, use CPU instead of GPU, or use the `-split` flag to process in chunks

- **Problem**: Audio format not supported
- **Solution**: Convert audio to WAV format using FFmpeg:
  ```bash
  ffmpeg -i input.mp3 input.wav
  ```

- **Problem**: `ModuleNotFoundError: No module named 'rvc.lib.tools.noisereduce'`
- **Solution**: This bug has been fixed. Update to the latest version.

- **Problem**: CREPE or legacy f0 methods silently produce wrong results
- **Solution**: This bug has been fixed. Update to the latest version.

- **Problem**: Autotune has no effect
- **Solution**: This bug has been fixed. The default `f0_autotune_strength` is now `1.0`. Make sure you're using the latest version.

#### NumPy Compatibility Issues
- **Problem**: `AttributeError: module 'numpy' has no attribute 'rint'`
- **Solution**: Upgrade to the latest version of this package which uses `np.round()` instead

### Performance Tips
- Use GPU acceleration when available for faster processing
- Process audio in smaller chunks for large files using `-split`
- Choose appropriate pitch extraction methods based on quality/speed requirements:
  - Fastest: PM, DIO
  - Balanced: Harvest, SWIPE
  - Highest Quality: RMVPE, CREPE (larger models)
- Use `-hl 128` for higher-quality pitch extraction (default is `64`; increase to trade speed for accuracy)

## Contributing

### Development Setup
1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/rvc.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate the environment: `source venv/bin/activate` (Linux/macOS) or `venv\Scripts\activate` (Windows)
5. Install in development mode: `pip install -e .`

### Code Standards
- Follow PEP 8 style guidelines
- Write docstrings for all public functions and classes
- Add type hints where appropriate
- Include tests for new functionality

### Pull Request Process
1. Create a feature branch: `git checkout -b feature/amazing-feature`
2. Make your changes
3. Add tests if applicable
4. Update documentation as needed
5. Submit a pull request to the main branch

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue on the GitHub repository with:
- Detailed description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, etc.)

## Acknowledgments

- Original Vietnamese RVC project for the foundational algorithm
- Various open-source libraries that make this project possible
- The community for contributions and feedback
