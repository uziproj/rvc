import os
import gc
import sys
import logging
import torch
import librosa
import requests
import numpy as np
import soundfile as sf
import torch.nn.functional as F
from tqdm import tqdm

# Use the centralized RVC logger instead of configuring a custom handler.
# This fixes the duplicate-output bug where every log line appeared twice
# (once from utils' own StreamHandler, once from the root logger).
from rvc.lib.logging import get_logger

logger = get_logger("utils")

# Import RVC modules - these will work when the package is installed via pip
from rvc.lib.backend import opencl

def change_rms(source_audio, source_rate, target_audio, target_rate, rate):
    """Change RMS of target audio to match source audio"""
    logger.debug(f"Changing RMS: source_rate={source_rate}, target_rate={target_rate}, rate={rate}")
    
    try:
        # Calculate RMS for source audio
        source_rms = librosa.feature.rms(
            y=source_audio, 
            frame_length=source_rate // 2 * 2, 
            hop_length=source_rate // 2
        )
        source_rms_tensor = torch.from_numpy(source_rms).float().unsqueeze(0)
        source_rms_interp = F.interpolate(
            source_rms_tensor, 
            size=target_audio.shape[0], 
            mode="linear"
        ).squeeze()
        
        # Calculate RMS for target audio
        target_rms = librosa.feature.rms(
            y=target_audio, 
            frame_length=target_rate // 2 * 2, 
            hop_length=target_rate // 2
        )
        target_rms_tensor = torch.from_numpy(target_rms).float().unsqueeze(0)
        target_rms_interp = F.interpolate(
            target_rms_tensor, 
            size=target_audio.shape[0], 
            mode="linear"
        ).squeeze()
        
        # Apply RMS adjustment
        result = target_audio * (
            torch.pow(
                source_rms_interp, 
                1 - rate
            ) * torch.pow(
                torch.maximum(target_rms_interp, torch.zeros_like(target_rms_interp) + 1e-6), 
                rate - 1
            )
        ).numpy()
        
        logger.debug("RMS adjustment completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Failed to change RMS: {e}")
        raise

def clear_gpu_cache():
    """Clear GPU cache and perform garbage collection"""
    logger.debug("Clearing GPU cache and running garbage collection")
    
    try:
        gc.collect()
        
        cleared_something = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            cleared_something = True
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
            cleared_something = True
        elif opencl.is_available():
            opencl.pytorch_ocl.empty_cache()
            cleared_something = True
        
        # Demoted from INFO to DEBUG — this is called many times per
        # conversion (once per chunk + once per pipeline call + once per
        # cleanup), so logging at INFO produced dozens of identical lines.
        # Users who want the verbose output can set log_level="debug".
        if cleared_something:
            logger.debug("GPU cache cleared successfully")
        else:
            logger.debug("No GPU cache to clear (CPU mode)")
        
    except Exception as e:
        logger.error(f"Failed to clear GPU cache: {e}")
        raise

def HF_download_file(url, output_path=None):
    """Download file from HuggingFace with progress bar"""
    logger.info(f"Starting download from: {url}")
    
    try:
        # Clean and prepare URL
        url = url.replace("/blob/", "/resolve/").replace("?download=true", "").strip()
        
        # Determine output path
        if output_path is None:
            output_path = os.path.basename(url)
        elif os.path.isdir(output_path):
            output_path = os.path.join(output_path, os.path.basename(url))
        
        logger.debug(f"Output path determined: {output_path}")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            logger.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        # Download file
        response = requests.get(url, stream=True, timeout=300)
        
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            filename = os.path.basename(output_path)
            
            logger.info(f"Downloading {filename} ({total_size / 1024 / 1024:.2f} MB)")
            
            # Initialize progress bar
            progress_bar = tqdm(
                total=total_size, 
                unit='B', 
                unit_scale=True, 
                desc=filename,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
            )
            
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=10 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))
            
            progress_bar.close()
            logger.info(f"Download completed: {output_path}")
            return output_path
            
        else:
            error_msg = f"Failed to download. Status Code: {response.status_code}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
    except requests.exceptions.Timeout:
        logger.error(f"Download timeout for URL: {url}")
        raise
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise

def check_predictors(method):
    """Check and download predictor models if needed"""
    logger.debug(f"Checking predictor for method: {method}")
    
    def download(predictors):
        """Download predictor model"""
        PREDICTOR_MODEL = os.path.join(os.getcwd(), "assets", "models", predictors)
        logger.debug(f"Predictor model path: {PREDICTOR_MODEL}")
        
        if not os.path.exists(PREDICTOR_MODEL):
            logger.info(f"Predictor model not found, downloading: {predictors}")
            
            # Ensure directory exists
            predictor_dir = os.path.dirname(PREDICTOR_MODEL)
            if predictor_dir and not os.path.exists(predictor_dir):
                logger.info(f"Creating predictor directory: {predictor_dir}")
                os.makedirs(predictor_dir, exist_ok=True)
            
            # Download the model
            download_url = os.path.join(
                "https://huggingface.co/NeoPy/Ultimate-Models/resolve/main/predictors/", 
                predictors
            )
            HF_download_file(download_url, PREDICTOR_MODEL)
        else:
            logger.debug(f"Predictor model already exists: {PREDICTOR_MODEL}")
    
    # Map methods to model files
    model_dict = {
        **dict.fromkeys(["rmvpe", "rmvpe-legacy"], "rmvpe.pt"), 
        **dict.fromkeys(["djcm"], "djcm.pt"),
        **dict.fromkeys(["fcpe"], "fcpe.pt"), 
        **dict.fromkeys(["fcpe-legacy"], "fcpe_legacy.pt"), 
        **dict.fromkeys(["crepe-full", "mangio-crepe-full"], "crepe_full.pth"), 
        **dict.fromkeys(["crepe-large", "mangio-crepe-large"], "crepe_large.pth"), 
        **dict.fromkeys(["crepe-medium", "mangio-crepe-medium"], "crepe_medium.pth"), 
        **dict.fromkeys(["crepe-small", "mangio-crepe-small"], "crepe_small.pth"), 
        **dict.fromkeys(["crepe-tiny", "mangio-crepe-tiny"], "crepe_tiny.pth"), 
    }
    
    # Check if method is a hybrid
    if method.startswith("hybrid[") and method.endswith("]"):
        # Extract methods inside hybrid brackets
        hybrid_methods = method[7:-1].split("+")
        logger.info(f"Hybrid method detected: {method}, components: {hybrid_methods}")
        
        # Download each component model
        for hybrid_method in hybrid_methods:
            if hybrid_method in model_dict:
                model_file = model_dict[hybrid_method]
                logger.info(f"Downloading hybrid component: {hybrid_method} -> {model_file}")
                download(model_file)
            else:
                logger.warning(f"No predictor model mapping found for hybrid component: {hybrid_method}")
    
    # Handle single methods
    elif method in model_dict:
        model_file = model_dict[method]
        logger.info(f"Found predictor model mapping: {method} -> {model_file}")
        download(model_file)
    
    # Handle methods that might be part of a hybrid but passed individually
    elif "+" in method:
        # This handles cases where someone might pass "rmvpe+fcpe" without hybrid[] wrapper
        logger.info(f"Detected method with '+' separator: {method}")
        individual_methods = method.split("+")
        
        for individual_method in individual_methods:
            if individual_method in model_dict:
                model_file = model_dict[individual_method]
                logger.info(f"Downloading component: {individual_method} -> {model_file}")
                download(model_file)
            else:
                logger.warning(f"No predictor model mapping found for component: {individual_method}")
    
    else:
        logger.warning(f"No predictor model mapping found for method: {method}")

def check_embedders(hubert):
    """Check and download embedder models if needed"""
    logger.debug(f"Checking embedder: {hubert}")
    
    if hubert in ["contentvec_base", "hubert_base", "japanese_hubert_base", 
                  "korean_hubert_base", "chinese_hubert_base", 
                  "portuguese_hubert_base", "spin"]:
        
        hubert_file = hubert + ".pt"
        HUBERT_PATH = os.path.join(os.getcwd(), "assets", "models", hubert_file)
        logger.debug(f"Hubert model path: {HUBERT_PATH}")
        
        if not os.path.exists(HUBERT_PATH): 
            logger.info(f"Hubert model not found, downloading: {hubert_file}")
            
            # Ensure directory exists
            hubert_dir = os.path.dirname(HUBERT_PATH)
            if hubert_dir and not os.path.exists(hubert_dir):
                logger.info(f"Creating hubert directory: {hubert_dir}")
                os.makedirs(hubert_dir, exist_ok=True)
            
            # Download the model
            download_url = os.path.join(
                "https://huggingface.co/NeoPy/Ultimate-Models/resolve/main/embedders/fairseq/", 
                hubert_file
            )
            HF_download_file(download_url, HUBERT_PATH)
        else:
            logger.debug(f"Hubert model already exists: {HUBERT_PATH}")
    else:
        logger.debug(f"Hubert model {hubert} does not require download")

def load_audio(file, sample_rate=16000, formant_shifting=False, formant_qfrency=0.8, formant_timbre=0.8):
    """Load audio file with optional formant shifting"""
    logger.info(f"Loading audio: {file}")

    try:
        # Clean file path
        file = file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        logger.debug(f"Cleaned file path: {file}")

        # Check if file exists
        if not os.path.isfile(file):
            error_msg = f"Audio file not found: {file}"
            logger.error(error_msg)
            raise FileNotFoundError(f"[ERROR] {error_msg}")

        logger.debug(f"File exists, size: {os.path.getsize(file) / 1024:.2f} KB")

        # Load audio file
        try:
            logger.debug("Attempting to load with soundfile...")
            audio, sr = sf.read(file, dtype=np.float32)
            logger.debug(f"Loaded with soundfile: sr={sr}, shape={audio.shape}")
        except Exception as sf_error:
            logger.warning(f"Soundfile failed, trying librosa: {sf_error}")
            audio, sr = librosa.load(file, sr=None)
            logger.debug(f"Loaded with librosa: sr={sr}, shape={audio.shape}")

        # Convert to mono if needed
        if len(audio.shape) > 1:
            logger.debug(f"Converting stereo to mono, original shape: {audio.shape}")
            audio = librosa.to_mono(audio.T)
            logger.debug(f"Converted to mono, new shape: {audio.shape}")

        # Resample if needed
        if sr != sample_rate:
            logger.info(f"Resampling from {sr}Hz to {sample_rate}Hz")
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=sample_rate,
                res_type="soxr_vhq"
            )
            logger.debug(f"Resampled audio shape: {audio.shape}")

        # Apply formant shifting if requested
        if formant_shifting:
            logger.info("Applying formant shifting")
            logger.debug(f"Formant parameters: qfrency={formant_qfrency}, timbre={formant_timbre}")

            try:
                from rvc.lib.backend.stftpitchshift import StftPitchShift

                pitchshifter = StftPitchShift(1024, 32, sample_rate)
                audio = pitchshifter.shiftpitch(
                    audio,
                    factors=1,
                    quefrency=formant_qfrency * 1e-3,
                    distortion=formant_timbre
                )
                logger.debug("Formant shifting applied successfully")
            except Exception as formant_error:
                logger.error(f"Formant shifting failed: {formant_error}")
                raise

        logger.info(f"Audio loaded successfully: shape={audio.shape}, dtype={audio.dtype}")
        return audio.flatten()

    except Exception as e:
        logger.error(f"Error loading audio file: {e}", exc_info=True)
        raise RuntimeError(f"[ERROR] Error reading audio file: {e}")

class Autotune:
    def __init__(self, ref_freqs):
        """Initialize Autotune with reference frequencies"""
        logger.debug(f"Initializing Autotune with {len(ref_freqs)} reference frequencies")
        self.ref_freqs = ref_freqs
        self.note_dict = self.ref_freqs
        logger.debug("Autotune initialized")
    
    def autotune_f0(self, f0, f0_autotune_strength):
        """Apply autotune to pitch contour"""
        logger.info(f"Applying autotune with strength: {f0_autotune_strength}")
        logger.debug(f"Input f0 shape: {f0.shape}, non-zero values: {np.count_nonzero(f0)}")
        
        if f0_autotune_strength <= 0:
            logger.debug("Autotune strength is 0, returning original f0")
            return f0
        
        try:
            autotuned_f0 = np.zeros_like(f0)
            total_notes = len(f0)
            
            # Process each frequency
            for i, freq in enumerate(f0):
                if freq > 0:  # Only process non-zero frequencies
                    # Find closest note frequency
                    closest_note = min(self.note_dict, key=lambda x: abs(x - freq))
                    # Apply autotune with strength factor
                    autotuned_f0[i] = freq + (closest_note - freq) * f0_autotune_strength
                else:
                    autotuned_f0[i] = 0
            
            # Calculate statistics
            non_zero_f0 = f0[f0 > 0]
            non_zero_autotuned = autotuned_f0[autotuned_f0 > 0]
            
            if len(non_zero_f0) > 0:
                avg_change = np.mean(np.abs(non_zero_autotuned - non_zero_f0))
                max_change = np.max(np.abs(non_zero_autotuned - non_zero_f0))
                logger.debug(f"Autotune statistics: avg_change={avg_change:.2f}, max_change={max_change:.2f}")
            
            logger.info(f"Autotune applied successfully to {len(non_zero_f0)}/{total_notes} non-zero values")
            return autotuned_f0
            
        except Exception as e:
            logger.error(f"Failed to apply autotune: {e}")
            return f0  # Return original f0 on error
