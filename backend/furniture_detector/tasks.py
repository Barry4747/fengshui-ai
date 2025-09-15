from celery import shared_task
import time
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
import logging

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
def process_image(self, task_id, picture_id, path):
    # MOCK SOLUTION
    time.sleep(5)

    return {"status": "done", "file": path}
