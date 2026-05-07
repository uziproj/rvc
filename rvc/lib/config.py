import os
import sys
import torch

# Import RVC modules - these will work when the package is installed via pip
from rvc.lib.backend import opencl

PREDICTOR_MODEL = os.path.join(os.getcwd(), "assets", "models")


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances: instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance

@singleton
class Config:
    def __init__(self, cpu_mode=False, is_half=False, embedder_model="contentvec_base", f0_method="rmvpe"):
        self.device = "cuda:0" if torch.cuda.is_available() else ("ocl:0" if opencl.is_available() else "cpu")
        self.is_half = is_half
        self.gpu_mem = None
        self.cpu_mode = cpu_mode
        if cpu_mode: self.device = "cpu"
        self.embedder_model = embedder_model
        self.f0_method = f0_method

        # Load hubert and rmvpe models at config initialization
        self.hubert_model = None
        self.rmvpe_model = None
        self._load_hubert(embedder_model)
        self._load_rmvpe(f0_method)

    def _load_hubert(self, embedder_model):
        """Load hubert/embedder model into memory."""
        try:
            from rvc.utils import check_embedders
            from rvc.lib.embedders import fairseq

            check_embedders(embedder_model)

            embedder_model_path = os.path.join(os.getcwd(), "assets", "models", embedder_model + ".pt")
            if not os.path.exists(embedder_model_path):
                print(f"[WARNING] Hubert model not found at {embedder_model_path}, skipping load.")
                return

            model = fairseq.load_model(embedder_model_path).to(self.device).eval()
            self.hubert_model = model.half() if self.is_half else model.float()
            print(f"[INFO] Hubert model '{embedder_model}' loaded on {self.device}.")
        except Exception as e:
            print(f"[WARNING] Failed to load hubert model: {e}")
            self.hubert_model = None

    def _load_rmvpe(self, f0_method):
        """Load RMVPE predictor model into memory."""
        try:
            from rvc.utils import check_predictors
            from rvc.lib.predictor.rmvpe import RMVPE

            # Check and download if needed
            check_predictors(f0_method)

            rmvpe_model_path = os.path.join(PREDICTOR_MODEL, "rmvpe.pt")
            if not os.path.exists(rmvpe_model_path):
                print(f"[WARNING] RMVPE model not found at {rmvpe_model_path}, skipping load.")
                return

            self.rmvpe_model = RMVPE(rmvpe_model_path, is_half=self.is_half, device=self.device)
            print(f"[INFO] RMVPE model loaded on {self.device}.")
        except Exception as e:
            print(f"[WARNING] Failed to load RMVPE model: {e}")
            self.rmvpe_model = None

    # INDENTATION FIXED: This method must be inside the class
    def device_config(self):
        if not self.cpu_mode:
            if self.device.startswith("cuda"): 
                self.set_cuda_config()
            elif opencl.is_available(): 
                self.device = "ocl:0"
                # Set default memory for OpenCL to prevent None errors
                self.gpu_mem = 4 
            elif self.has_mps(): 
                self.device = "mps"
                # Set default memory for MPS to prevent None errors
                self.gpu_mem = 4 
            else: 
                self.device = "cpu"

        # Ensure gpu_mem is not None before checking logic
        if self.gpu_mem is not None and self.gpu_mem <= 4: 
            return 1, 5, 30, 32
        return (3, 10, 60, 65) if self.is_half else (1, 6, 38, 41)

    # INDENTATION FIXED
    def set_cuda_config(self):
        i_device = int(self.device.split(":")[-1])
        self.gpu_mem = torch.cuda.get_device_properties(i_device).total_memory // (1024**3)

    # INDENTATION FIXED
    def has_mps(self):
        return torch.backends.mps.is_available()
