"""
RVC REST API - FastAPI application for voice conversion.

Provides endpoints for loading voice models, converting audio files,
and managing the inference pipeline over HTTP.

Usage:
    python -m rvc.api.app
    # or
    uvicorn rvc.api.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from rvc.infer.cli import VoiceConverter
from rvc.lib.config import Config
from rvc.utils import check_embedders, check_predictors
from rvc.var import method as F0_METHODS

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rvc.api")

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
EMBEDDER_MODELS = [
    "contentvec_base",
    "hubert_base",
    "japanese_hubert_base",
    "korean_hubert_base",
    "chinese_hubert_base",
    "portuguese_hubert_base",
    "spin",
]

SUPPORTED_INPUT_FORMATS = {
    "wav", "mp3", "flac", "ogg", "opus", "m4a", "mp4",
    "aac", "alac", "wma", "aiff", "webm", "ac3",
}

SUPPORTED_OUTPUT_FORMATS = {"wav", "flac", "mp3", "ogg"}

OUTPUT_MIME_TYPES = {
    "wav": "audio/wav",
    "flac": "audio/flac",
    "mp3": "audio/mpeg",
    "ogg": "audio/ogg",
}

# Model registry: model_id -> VoiceConverter
_model_registry: dict[str, VoiceConverter] = {}

# Config singleton (created once at startup)
_config: Config | None = None

# Temp directory for uploaded/converted files
_tmp_dir = tempfile.mkdtemp(prefix="rvc_api_")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RVC Voice Conversion API",
    description=(
        "REST API for Retrieval-based Voice Conversion (RVC). "
        "Load voice models, convert audio, and manage inference over HTTP."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class ModelLoadRequest(BaseModel):
    """Request body for loading a voice model."""
    model_path: str = Field(..., description="Absolute path to the .pth voice model file")
    sid: int = Field(0, description="Speaker ID (for multi-speaker models)")


class ModelLoadResponse(BaseModel):
    model_id: str
    model_path: str
    version: str | None = None
    vocoder: str | None = None
    use_f0: bool | None = None
    n_spk: int | None = None
    target_sr: int | None = None


class ConvertRequest(BaseModel):
    """Request body for path-based audio conversion."""
    model_id: str = Field(..., description="ID of the loaded voice model")
    input_path: str = Field(..., description="Path to input audio file on the server")
    output_format: str = Field("wav", description="Output format: wav, flac, mp3, ogg")
    pitch: int = Field(0, description="Pitch shift in semitones")
    f0_method: str = Field("rmvpe", description="F0 extraction method")
    embedder_model: str = Field("contentvec_base", description="Embedder model name")
    index_path: str | None = Field(None, description="Path to .index FAISS index file")
    index_rate: float = Field(0.5, description="Feature retrieval blend ratio [0..1]", ge=0.0, le=1.0)
    filter_radius: int = Field(3, description="Median filter radius for F0", ge=0)
    volume_envelope: float = Field(1.0, description="RMS envelope mixing ratio", ge=0.0, le=1.0)
    protect: float = Field(0.5, description="Voiceless consonant protection [0..1]", ge=0.0, le=0.5)
    hop_length: int = Field(64, description="Hop length for F0 extraction", ge=1)
    resample_sr: int = Field(0, description="Target sample rate (0 = no resampling)", ge=0)
    f0_autotune: bool = Field(False, description="Enable pitch autotune")
    f0_autotune_strength: float = Field(1.0, description="Autotune strength [0..1]", ge=0.0, le=1.0)
    split_audio: bool = Field(False, description="Split audio at silence for processing")
    clean_audio: bool = Field(False, description="Apply noise reduction to output")
    clean_strength: float = Field(0.7, description="Noise reduction strength [0..1]", ge=0.0, le=1.0)
    formant_shifting: bool = Field(False, description="Enable formant shifting")
    formant_qfrency: float = Field(0.8, description="Formant quefrency", ge=0.0)
    formant_timbre: float = Field(0.8, description="Formant timbre", ge=0.0)
    proposal_pitch: bool = Field(False, description="Auto-calculate optimal pitch shift")
    proposal_pitch_threshold: float = Field(255.0, description="Target pitch for proposal", ge=0.0)


class ConvertUploadParams(BaseModel):
    """Query parameters for file-upload based conversion."""
    model_id: str = Field(..., description="ID of the loaded voice model")
    output_format: str = Field("wav", description="Output format: wav, flac, mp3, ogg")
    pitch: int = Field(0, description="Pitch shift in semitones")
    f0_method: str = Field("rmvpe", description="F0 extraction method")
    embedder_model: str = Field("contentvec_base", description="Embedder model name")
    index_path: str | None = Field(None, description="Path to .index FAISS index file")
    index_rate: float = Field(0.5, description="Feature retrieval blend ratio")
    filter_radius: int = Field(3, description="Median filter radius for F0")
    volume_envelope: float = Field(1.0, description="RMS envelope mixing ratio")
    protect: float = Field(0.5, description="Voiceless consonant protection")
    hop_length: int = Field(64, description="Hop length for F0 extraction")
    resample_sr: int = Field(0, description="Target sample rate (0 = no resampling)")
    f0_autotune: bool = Field(False, description="Enable pitch autotune")
    f0_autotune_strength: float = Field(1.0, description="Autotune strength")
    split_audio: bool = Field(False, description="Split audio at silence")
    clean_audio: bool = Field(False, description="Apply noise reduction")
    clean_strength: float = Field(0.7, description="Noise reduction strength")
    formant_shifting: bool = Field(False, description="Enable formant shifting")
    formant_qfrency: float = Field(0.8, description="Formant quefrency")
    formant_timbre: float = Field(0.8, description="Formant timbre")
    proposal_pitch: bool = Field(False, description="Auto-calculate optimal pitch")
    proposal_pitch_threshold: float = Field(255.0, description="Target pitch for proposal")


class ConvertResponse(BaseModel):
    """Response for path-based conversion."""
    success: bool
    output_path: str | None = None
    message: str | None = None


class ErrorResponse(BaseModel):
    detail: str


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    global _config
    logger.info("Initializing RVC Config...")
    _config = Config(embedder_model="contentvec_base", f0_method="rmvpe")
    logger.info(f"Device: {_config.device}, Half precision: {_config.is_half}")
    os.makedirs(os.path.join(os.getcwd(), "assets", "models"), exist_ok=True)
    logger.info("RVC API ready.")


@app.on_event("shutdown")
async def shutdown():
    # Clean up all loaded models
    for mid, cvt in list(_model_registry.items()):
        try:
            cvt.cleanup()
        except Exception:
            pass
    _model_registry.clear()
    # Remove temp directory
    shutil.rmtree(_tmp_dir, ignore_errors=True)
    logger.info("RVC API shut down.")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _generate_model_id(model_path: str) -> str:
    """Generate a deterministic model ID from the model file path."""
    abs_path = os.path.abspath(model_path)
    return hashlib.md5(abs_path.encode()).hexdigest()[:12]


def _get_converter(model_id: str) -> VoiceConverter:
    """Retrieve a VoiceConverter from the registry or raise 404."""
    if model_id not in _model_registry:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not loaded. Use POST /api/v1/models/load first.")
    return _model_registry[model_id]


def _validate_f0_method(f0_method: str) -> None:
    """Validate F0 method name."""
    if f0_method.startswith("hybrid[") and f0_method.endswith("]"):
        return  # Hybrid methods are valid
    if f0_method not in F0_METHODS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid F0 method '{f0_method}'. Available: {F0_METHODS} or hybrid[method1+method2]",
        )


def _validate_output_format(fmt: str) -> None:
    """Validate output format."""
    if fmt not in SUPPORTED_OUTPUT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid output format '{fmt}'. Supported: {sorted(SUPPORTED_OUTPUT_FORMATS)}",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/health", tags=["System"])
async def health_check():
    """Check API health and system status."""
    return {
        "status": "healthy",
        "device": _config.device if _config else "unknown",
        "models_loaded": len(_model_registry),
    }


@app.get("/api/v1/config", tags=["System"])
async def get_config():
    """Get current server configuration."""
    if not _config:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return {
        "device": _config.device,
        "is_half": _config.is_half,
        "cpu_mode": _config.cpu_mode,
        "gpu_mem": _config.gpu_mem,
    }


@app.get("/api/v1/methods", tags=["Reference"])
async def list_f0_methods():
    """List available F0 (pitch) extraction methods."""
    return {
        "methods": F0_METHODS,
        "hybrid_example": "hybrid[rmvpe+fcpe]",
        "count": len(F0_METHODS),
    }


@app.get("/api/v1/embedders", tags=["Reference"])
async def list_embedders():
    """List available embedder models."""
    return {
        "embedders": EMBEDDER_MODELS,
        "count": len(EMBEDDER_MODELS),
    }


# --- Model Management ---

@app.post(
    "/api/v1/models/load",
    response_model=ModelLoadResponse,
    tags=["Models"],
    summary="Load a voice model",
    description="Load a .pth voice model into memory for conversion. Returns a model_id for use in conversion requests.",
)
async def load_model(req: ModelLoadRequest):
    """Load a voice model (.pth) into the registry."""
    # Validate path
    if not os.path.exists(req.model_path):
        raise HTTPException(status_code=404, detail=f"Model file not found: {req.model_path}")
    if not req.model_path.endswith(".pth"):
        raise HTTPException(status_code=400, detail="Model file must be a .pth file")
    if os.path.isdir(req.model_path):
        raise HTTPException(status_code=400, detail="Path is a directory, not a file")

    model_id = _generate_model_id(req.model_path)

    # If already loaded, return existing info
    if model_id in _model_registry:
        cvt = _model_registry[model_id]
        return ModelLoadResponse(
            model_id=model_id,
            model_path=req.model_path,
            version=cvt.version,
            vocoder=cvt.vocoder,
            use_f0=bool(cvt.use_f0),
            n_spk=cvt.n_spk,
            target_sr=cvt.tgt_sr,
        )

    # Load model (blocking, run in thread)
    try:
        cvt = await asyncio.to_thread(VoiceConverter, _config, req.model_path, req.sid)
    except Exception as e:
        logger.exception(f"Failed to load model: {req.model_path}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    _model_registry[model_id] = cvt
    logger.info(f"Loaded model '{model_id}' from {req.model_path}")

    return ModelLoadResponse(
        model_id=model_id,
        model_path=req.model_path,
        version=cvt.version,
        vocoder=cvt.vocoder,
        use_f0=bool(cvt.use_f0),
        n_spk=cvt.n_spk,
        target_sr=cvt.tgt_sr,
    )


@app.get("/api/v1/models", tags=["Models"])
async def list_models():
    """List all loaded voice models."""
    models = []
    for mid, cvt in _model_registry.items():
        models.append({
            "model_id": mid,
            "loaded_model": cvt.loaded_model,
            "version": cvt.version,
            "vocoder": cvt.vocoder,
            "use_f0": bool(cvt.use_f0),
            "n_spk": cvt.n_spk,
            "target_sr": cvt.tgt_sr,
        })
    return {"models": models, "count": len(models)}


@app.get("/api/v1/models/{model_id}", tags=["Models"])
async def get_model_info(model_id: str):
    """Get info about a specific loaded model."""
    cvt = _get_converter(model_id)
    return {
        "model_id": model_id,
        "loaded_model": cvt.loaded_model,
        "version": cvt.version,
        "vocoder": cvt.vocoder,
        "use_f0": bool(cvt.use_f0),
        "n_spk": cvt.n_spk,
        "target_sr": cvt.tgt_sr,
        "device": cvt.device,
    }


@app.delete("/api/v1/models/{model_id}", tags=["Models"])
async def unload_model(model_id: str):
    """Unload a voice model from memory."""
    cvt = _get_converter(model_id)
    try:
        cvt.cleanup()
    except Exception as e:
        logger.warning(f"Error during model cleanup: {e}")
    del _model_registry[model_id]
    logger.info(f"Unloaded model '{model_id}'")
    return {"message": f"Model '{model_id}' unloaded successfully"}


# --- Audio Conversion ---

@app.post(
    "/api/v1/convert",
    response_class=StreamingResponse,
    tags=["Conversion"],
    summary="Convert audio (file upload)",
    description=(
        "Upload an audio file for voice conversion. "
        "Returns the converted audio as a streaming response. "
        "This is the recommended endpoint for remote clients."
    ),
)
async def convert_upload(
    audio: UploadFile = File(..., description="Input audio file (wav, mp3, flac, ogg, etc.)"),
    model_id: str = "default",
    output_format: str = "wav",
    pitch: int = 0,
    f0_method: str = "rmvpe",
    embedder_model: str = "contentvec_base",
    index_path: Optional[str] = None,
    index_rate: float = 0.5,
    filter_radius: int = 3,
    volume_envelope: float = 1.0,
    protect: float = 0.5,
    hop_length: int = 64,
    resample_sr: int = 0,
    f0_autotune: bool = False,
    f0_autotune_strength: float = 1.0,
    split_audio: bool = False,
    clean_audio: bool = False,
    clean_strength: float = 0.7,
    formant_shifting: bool = False,
    formant_qfrency: float = 0.8,
    formant_timbre: float = 0.8,
    proposal_pitch: bool = False,
    proposal_pitch_threshold: float = 255.0,
):
    """Convert an uploaded audio file and return the result as streaming audio."""
    # Validate
    _validate_f0_method(f0_method)
    _validate_output_format(output_format)
    cvt = _get_converter(model_id)

    # Save uploaded file to temp dir
    ext = Path(audio.filename).suffix.lstrip(".") if audio.filename else "wav"
    if ext not in SUPPORTED_INPUT_FORMATS:
        ext = "wav"

    uid = uuid.uuid4().hex[:8]
    input_path = os.path.join(_tmp_dir, f"input_{uid}.{ext}")
    output_path = os.path.join(_tmp_dir, f"output_{uid}.{output_format}")

    try:
        # Write uploaded file
        with open(input_path, "wb") as f:
            content = await audio.read()
            f.write(content)

        logger.info(f"Converting: {input_path} -> {output_path} (model={model_id}, pitch={pitch}, f0={f0_method})")

        # Check and download predictor/embedder models if needed
        await asyncio.to_thread(check_predictors, f0_method)
        await asyncio.to_thread(check_embedders, embedder_model)

        # Run conversion in thread pool (blocking operation)
        await asyncio.to_thread(
            cvt.convert_audio,
            audio_input_path=input_path,
            audio_output_path=output_path,
            index_path=index_path or "",
            embedder_model=embedder_model,
            pitch=pitch,
            f0_method=f0_method,
            index_rate=index_rate,
            volume_envelope=volume_envelope,
            protect=protect,
            hop_length=hop_length,
            filter_radius=filter_radius,
            export_format=output_format,
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

        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Conversion produced no output file")

        # Stream the output file back
        mime = OUTPUT_MIME_TYPES.get(output_format, "application/octet-stream")
        file_size = os.path.getsize(output_path)

        async def _stream():
            with open(output_path, "rb") as f:
                while True:
                    chunk = f.read(64 * 1024)
                    if not chunk:
                        break
                    yield chunk
            # Cleanup temp files after streaming
            for p in (input_path, output_path):
                try:
                    os.remove(p)
                except OSError:
                    pass

        return StreamingResponse(
            _stream(),
            media_type=mime,
            headers={
                "Content-Disposition": f'attachment; filename="converted_{uid}.{output_format}"',
                "Content-Length": str(file_size),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Conversion failed: {e}")
        # Cleanup on error
        for p in (input_path, output_path):
            try:
                os.remove(p)
            except OSError:
                pass
        raise HTTPException(status_code=500, detail=f"Conversion failed: {e}")


@app.post(
    "/api/v1/convert/file",
    response_model=ConvertResponse,
    tags=["Conversion"],
    summary="Convert audio (server path)",
    description=(
        "Convert an audio file that already exists on the server filesystem. "
        "Returns the output file path. Use this when both input and output "
        "files are accessible on the server."
    ),
)
async def convert_file(req: ConvertRequest):
    """Convert audio from a server file path and save to a server path."""
    _validate_f0_method(req.f0_method)
    _validate_output_format(req.output_format)
    cvt = _get_converter(req.model_id)

    if not os.path.exists(req.input_path):
        raise HTTPException(status_code=404, detail=f"Input file not found: {req.input_path}")

    # Determine output path
    base, _ = os.path.splitext(req.input_path)
    output_path = f"{base}_converted.{req.output_format}"

    try:
        logger.info(f"Converting: {req.input_path} -> {output_path} (model={req.model_id})")

        # Check and download models if needed
        await asyncio.to_thread(check_predictors, req.f0_method)
        await asyncio.to_thread(check_embedders, req.embedder_model)

        # Run conversion
        await asyncio.to_thread(
            cvt.convert_audio,
            audio_input_path=req.input_path,
            audio_output_path=output_path,
            index_path=req.index_path or "",
            embedder_model=req.embedder_model,
            pitch=req.pitch,
            f0_method=req.f0_method,
            index_rate=req.index_rate,
            volume_envelope=req.volume_envelope,
            protect=req.protect,
            hop_length=req.hop_length,
            filter_radius=req.filter_radius,
            export_format=req.output_format,
            resample_sr=req.resample_sr,
            f0_autotune=req.f0_autotune,
            f0_autotune_strength=req.f0_autotune_strength,
            split_audio=req.split_audio,
            clean_audio=req.clean_audio,
            clean_strength=req.clean_strength,
            formant_shifting=req.formant_shifting,
            formant_qfrency=req.formant_qfrency,
            formant_timbre=req.formant_timbre,
            proposal_pitch=req.proposal_pitch,
            proposal_pitch_threshold=req.proposal_pitch_threshold,
        )

        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Conversion produced no output file")

        return ConvertResponse(
            success=True,
            output_path=output_path,
            message=f"Conversion complete. Output saved to {output_path}",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Conversion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Conversion failed: {e}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    """Run the API server."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="RVC Voice Conversion REST API")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--workers", type=int, default=1, help="Number of Uvicorn workers (default: 1)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    uvicorn.run(
        "rvc.api.app:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
