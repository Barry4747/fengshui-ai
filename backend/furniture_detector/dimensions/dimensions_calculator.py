
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ..depth.model.registry import ModelManager

def segment_with_boxes(image, detected_boxes):
    model = ModelManager.get_model('sam-vit-h')
    return model.auto_segment_with_boxes(image, detected_boxes)


def mask_iou(mask: np.ndarray, bbox: list) -> float:
    """
    Compute IoU between a binary mask and a bounding box.
    """
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return 0.0
    mx1, my1, mx2, my2 = xs.min(), ys.min(), xs.max(), ys.max()
    bx1, by1, bx2, by2 = bbox
    ix1 = max(mx1, bx1)
    iy1 = max(my1, by1)
    ix2 = min(mx2, bx2)
    iy2 = min(my2, by2)
    iw = max(ix2 - ix1, 0)
    ih = max(iy2 - iy1, 0)
    inter = iw * ih
    union = (mx2 - mx1) * (my2 - my1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0

def select_best_mask_for_bbox(bbox: list, masks: list) -> dict:
    """
    Choose the SAM mask with the highest IoU for a given bounding box.
    """
    best_mask = None
    best_iou = 0.0
    for mask in masks:
        iou = mask_iou(mask["segmentation"], bbox)
        if iou > best_iou:
            best_iou = iou
            best_mask = mask
    return best_mask


def compute_object_dimensions(segmentation: np.ndarray, bbox: list, depth_map: np.ndarray) -> dict:
    """
    Compute width, height, and depth of a single object from its segmentation mask and depth map.
    Returns pixel-based and optionally scaled measurements with points for visualization.
    """
    segmentation = segmentation.astype(bool)
    masked_depth = depth_map * segmentation
    object_depth_values = masked_depth[masked_depth > 0]

    if len(object_depth_values) == 0:
        return None

    depth_mean = float(object_depth_values.mean())

    ys, xs = np.where(segmentation)
    pixel_height = ys.max() - ys.min()
    pixel_width = xs.max() - xs.min()
    pixel_depth = object_depth_values.max() - object_depth_values.min()  # length along Z

    width_points = ((xs.min(), ys.mean()), (xs.max(), ys.mean()))
    height_points = ((xs.mean(), ys.min()), (xs.mean(), ys.max()))
    depth_points = ((xs.mean(), ys.mean(), object_depth_values.min()), (xs.mean(), ys.mean(), object_depth_values.max()))

    
    width_m, height_m, depth_m = pixel_width, pixel_height, pixel_depth

    return {
        "bbox": bbox,
        "depth_mean": depth_mean,
        "width": width_m,
        "height": height_m,
        "depth": depth_m,
        "width_points": width_points,
        "height_points": height_points,
        "depth_points": depth_points
    }


def draw_debug_dimensions(image: Image.Image, object_dims: list) -> Image.Image:
    """
    Draw lines on the image showing width, height, and depth.
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for dims in object_dims:
        # Width line
        x1, y1 = dims["width_points"][0]
        x2, y2 = dims["width_points"][1]
        draw.line([x1, y1, x2, y2], fill="red", width=2)
        draw.text((x1, y1 - 10), f"W: {dims['width']:.1f}", fill="red", font=font)

        # Height line
        x1, y1 = dims["height_points"][0]
        x2, y2 = dims["height_points"][1]
        draw.line([x1, y1, x2, y2], fill="green", width=2)
        draw.text((x1 - 40, y1), f"H: {dims['height']:.1f}", fill="green", font=font)

        # Depth annotation
        draw.text((dims["bbox"][0], dims["bbox"][1] - 25), f"D: {dims['depth']:.1f}", fill="blue", font=font)

    return image
