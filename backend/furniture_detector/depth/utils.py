from PIL import Image
import numpy as np

def save_depth_with_pil(depth: np.ndarray, out_path: str = "depth_gray.png"):
    depth_min = np.min(depth)
    depth_max = np.max(depth)
    depth_norm = (255 * (depth - depth_min) / (depth_max - depth_min)).astype(np.uint8)

    depth_img = Image.fromarray(depth_norm, mode="L")
    #depth_img.save(out_path)
    return depth_img
