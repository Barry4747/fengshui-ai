from celery import shared_task
import time
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
import logging
from .depth.model.registry import ModelManager
from .models import Picture, Task
from .dimensions.dimensions_calculator import segment_with_boxes, select_best_mask_for_bbox, compute_object_dimensions, draw_debug_dimensions
from .depth.depth_calculator import get_depth_image
from PIL import Image, ImageDraw, ImageFont
import numpy as np


logger = logging.getLogger(__name__)


def send_progress(session_id, event_type, **kwargs):
    """Send real-time progress update via WebSocket."""
    try:
        channel_layer = get_channel_layer()
        async_to_sync(channel_layer.group_send)(
            f"progress_{session_id}",
            {
                "type": "task.progress",
                "event": event_type,
                **kwargs
            }
        )
    except Exception as e:
        logger.error(f"Failed to send progress update: {str(e)}")


def update_job_status(task, status, session_id=None, **kwargs):
    """Update task status, and broadcast progress."""
    task.status = status
    
    if session_id:
        send_progress(session_id, status, task_id=task.id, **kwargs)


def serialize_dimensions(dims):
    """
    Convert all numpy types to native Python types to be JSON serializable.
    """
    return {
        "class_id": int(dims["class_id"]),
        "bbox": [int(x) for x in dims["bbox"]],
        "depth_mean": float(dims["depth_mean"]),
        "width": float(dims["width"]),
        "height": float(dims["height"]),
        "depth": float(dims["depth"]),
        # Optional: convert points to tuples
        "width_points": [(int(x), int(y)) for x, y in dims.get("width_points", [])],
        "height_points": [(int(x), int(y)) for x, y in dims.get("height_points", [])],
        "depth_points": [(float(x), float(y), float(z)) for x, y, z in dims.get("depth_points", [])]
    }



@shared_task(bind=True)
def process_image(self, task_id, picture_id, session_id, path, model_name='furniture_yolo'):
    task = Task.objects.get(id=task_id)
    picture = Picture.objects.get(id=picture_id)

    if not task:
        raise ValueError(f"No task labeled with id: {task_id}")
    
    if not picture:
        raise ValueError(f"No picture labeled with id: {picture_id}")

    update_job_status(task, "decoding", session_id=session_id)

    model = ModelManager.get_model(model_name=model_name, model_category='detection')    

    results = model.predict_image(path)

    picture.detected_data = results
    picture.save()

    update_job_status(task, "decoding_finished", session_id=session_id)

    return {"status": "done", "results": results}


@shared_task(bind=True)
def calculate_dimensions(self, session_id, task_id, picture_id):
    """
    Main task: calculate object dimensions using detected bounding boxes and SAM masks.
    Debug version: draws width/height lines directly on the image.
    """

    picture = Picture.objects.get(id=picture_id)
    data = picture.detected_data  # YOLO detections

    image = Image.open(picture.image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    masks = segment_with_boxes(image, data)  # SAM masks
    depth_map = get_depth_image(picture.image_path)

    object_dimensions = []

    for det in data:
        bbox = det["bbox"]
        class_id = det["class_id"]

        best_mask = select_best_mask_for_bbox(bbox, masks)
        if best_mask is None:
            continue

        dims = compute_object_dimensions(best_mask["segmentation"], bbox, depth_map)
        if dims is None:
            continue

        dims["class_id"] = class_id

        object_dimensions.append(serialize_dimensions(dims))

    debug_image = draw_debug_dimensions(image.copy(), object_dimensions)
    debug_image.show()

    print(object_dimensions)
    return {"status": "done", "results": object_dimensions}
    



