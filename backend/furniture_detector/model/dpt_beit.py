import torch
import cv2
import os
import sys
from PIL import Image 
from .base_model_for_registry import BaseModel
from .midas.transforms import PrepareForNet, NormalizeImage
import tempfile
import numpy as np
from torchvision.transforms import Compose, Resize, InterpolationMode, Normalize

def get_dpt_transform():
    """Returns transform pipeline compatible with DPT BEiT 512x512."""
    transform = Compose(
        [
            Resize(size=512, interpolation=InterpolationMode.BICUBIC),
            lambda x: torch.from_numpy(np.array(x)).permute(2, 0, 1).float() / 255.0,
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    def debug_transform(x):
        if isinstance(x, Image.Image):
            w, h = x.size
            mode = x.mode
            x_np = np.array(x)
            x_min, x_max = x_np.min(), x_np.max()
            print(f"[TRANSFORM] Input PIL image: size=({w}, {h}), mode={mode}, range=[{x_min:.1f}, {x_max:.1f}]")
        elif isinstance(x, np.ndarray):
            print(f"[TRANSFORM] Input numpy array: shape={x.shape}, dtype={x.dtype}, range=[{x.min():.1f}, {x.max():.1f}]")
        elif isinstance(x, torch.Tensor):
            print(f"[TRANSFORM] Input tensor: shape={x.shape}, dtype={x.dtype}, range=[{x.min().item():.3f}, {x.max().item():.3f}]")
        else:
            print(f"[TRANSFORM] Input type: {type(x)}")

        out = transform(x)

        if isinstance(out, torch.Tensor):
            print(f"[TRANSFORM] Output tensor shape: {out.shape}, dtype: {out.dtype}, range: [{out.min().item():.3f}, {out.max().item():.3f}]")
        else:
            print(f"[TRANSFORM] Output type: {type(out)}")

        return out

    return debug_transform


class MiDaSModel(BaseModel):
    """
    MiDaS v3 depth estimation model wrapper compatible with ModelManager.
    Computes relative depth from a single image.
    """

    def __init__(self, device: int = 0):
        """
        device: GPU device index (int) or -1 for CPU
        """
        self.device = torch.device(
            f"cuda:{device}" if torch.cuda.is_available() and device >= 0 else "cpu"
        )
        self.model = None
        self.transform = None
        self.model_type = None

    def load_model(
        self, model_path: str = None, model_type: str = "DPT_Large", repo_path: str = None, **kwargs
    ):
        """
        Load MiDaS model from local path.
        Supports BEiT checkpoint (dpt_beit_large_512.pt).
        """
        self.model_type = model_type

        if repo_path and repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        from midas.dpt_depth import DPTDepthModel

        if model_path and "beit_large_512" in os.path.basename(model_path).lower():
            print(f"[MiDaSModel] Loading BEiT model from {model_path}")

            checkpoint = torch.load(model_path, map_location="cpu")

            unexpected_keys = [k for k in checkpoint.keys() if "relative_position_index" in k]
            if unexpected_keys:
                print(f"[WARNING] Removing unexpected keys: {unexpected_keys}")
                for k in unexpected_keys:
                    del checkpoint[k]

            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
                torch.save(checkpoint, tmp_file.name)
                clean_model_path = tmp_file.name

            self.transform = get_dpt_transform()
            self.model = DPTDepthModel(
                path=clean_model_path,
                backbone="beitl16_512",
                non_negative=True,
            )

            os.unlink(clean_model_path)

        else:
            raise NotImplementedError("Obsługiwany jest tylko model dpt_beit_large_512.pt")

        self.model.to(self.device)
        self.model.eval()
        print(f"[MiDaSModel] Loaded {self.model_type} on device {self.device}")
    
    
    def unload_model(self):
        """
        Free GPU memory used by MiDaS.
        """
        if self.model is not None:
            del self.model
            del self.transform
            torch.cuda.empty_cache()
            self.model = None
            self.transform = None
            print(f"[MiDaSModel] Unloaded {self.model_type}")

    def predict_image(self, image_path: str):
        if self.model is None or self.transform is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path} — may be corrupted or unsupported format.")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_pil = Image.fromarray(img.astype('uint8'), 'RGB')

        input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
        print('[MiDaSModel] Running inference...')
        print(f"Model device: {next(self.model.parameters()).device}")
        print(f"Input tensor device: {input_tensor.device}")
        with torch.no_grad():
            import traceback

            with torch.no_grad():
                try:
                    print('[MiDaSModel] Przed wywołaniem modelu...')
                    prediction = self.model(input_tensor)
                    print('[MiDaSModel] Model prediction complete.')
                except Exception as e:
                    print('[MiDaSModel] Błąd podczas wywołania modelu:')
                    traceback.print_exc()
                    raise e
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2], 
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        print('[MiDaSModel] Inference done.')
        return prediction.cpu().numpy()

    def predict_folder(self, folder_path: str):
        """
        Run depth prediction on all images in a folder.
        Returns dict: {filename: depth_map (numpy array)}
        """

        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"Folder not found: {folder_path}")

        results = {}
        supported_ext = (".png", ".jpg", ".jpeg")
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(supported_ext):
                path = os.path.join(folder_path, fname)
                try:
                    results[fname] = self.predict_image(path)
                    print(f"[MiDaSModel] Processed: {fname}")
                except Exception as e:
                    print(f"[MiDaSModel] Error processing {fname}: {e}")
                    continue

        return results