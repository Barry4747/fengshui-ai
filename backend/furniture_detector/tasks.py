from celery import shared_task
import time
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
import logging
from .model.registry import ModelManager
from .models import Task, Picture

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
