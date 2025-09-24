import torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import logging
from PIL import Image
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

def is_number(s):
    try:
        float(s)
        return True
    except:
        return False


class SAMSegmenter:
    """
    Automatic segmentation using Meta's Segment Anything Model (SAM).
    Can be used standalone or combined with object detection results.
    """

    def __init__(self, model_type: str = "vit_h", device: Optional[str] = None):
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sam = None
        self.mask_generator = None

    # ---------------- Model Loading / Unloading ----------------

    def load_model(self, checkpoint_path: str):
        """
        Load the SAM model from a checkpoint file.
        checkpoint_path: path to the .pth weights
        """
        try:
            logger.info(f"Loading SAM model '{self.model_type}' from {checkpoint_path} on {self.device}")
            self.sam = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
            self.sam.to(device=self.device)
            self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        except Exception as e:
            logger.exception(f"Exception occurred while loading SAM: {e}")
            raise e

    def unload_model(self):
        """
        Unload the SAM model to free GPU memory.
        """
        try:
            if self.sam is not None:
                del self.sam
                self.sam = None
            self.mask_generator = None
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
            logger.info("SAM model unloaded successfully")
        except Exception as e:
            logger.warning(f"Error while unloading SAM: {e}")

    # ---------------- Segmentation ----------------

    def auto_segment(self, image: Image.Image) -> List[Dict]:
        """
        Perform automatic segmentation on an RGB image.
        image: PIL.Image.Image
        returns: List of masks in SAM's dictionary format
        """
        if self.mask_generator is None:
            raise RuntimeError("SAM model is not loaded. Call load_model() first.")

        try:
            # Ensure image is RGB
            if image.mode != "RGB":
                image = image.convert("RGB")
            np_image = np.array(image)

            # Generate masks using SAM
            masks = self.mask_generator.generate(np_image)
            return masks
        except Exception as e:
            logger.exception(f"Error while segmenting image: {e}")
            return []

    def auto_segment_with_boxes(self, image: Image.Image, boxes: list):
        """
        Segment objects using SAM within YOLO-provided bounding boxes.
        
        image: PIL.Image.Image (RGB)
        boxes: list of dicts, each with keys:
            - "class_id": int
            - "confidence": float
            - "bbox": [x1, y1, x2, y2] (all floats)
        
        returns: list of SAM masks aligned with the original image
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        np_image = np.array(image)
        h, w = np_image.shape[:2]

        all_masks = []

        for box in boxes:
            try:
                # Extract bbox coordinates
                x1, y1, x2, y2 = map(int, box["bbox"])

                # Clip to image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 <= x1 or y2 <= y1:
                    # Skip invalid boxes
                    continue

                # Crop patch
                patch = np_image[y1:y2, x1:x2]

                # Generate masks on the patch
                patch_masks = self.mask_generator.generate(patch)

                # Shift masks coordinates to full image
                for mask in patch_masks:
                    seg = mask["segmentation"]
                    # Create full-size mask
                    full_mask = np.zeros((h, w), dtype=seg.dtype)
                    full_mask[y1:y2, x1:x2] = seg
                    mask["segmentation"] = full_mask
                    mask["bbox"] = [x1, y1, x2, y2]
                    mask["class_id"] = box["class_id"]
                    mask["confidence"] = box["confidence"]

                all_masks.extend(patch_masks)

            except Exception as e:
                logger.warning(f"Skipping box {box} due to error: {e}")
                continue

        return all_masks
