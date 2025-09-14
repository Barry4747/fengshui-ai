from celery import shared_task
import time

@shared_task(bind=True)
def process_image(self, task_id, path):
    # MOCK SOLUTION
    time.sleep(5)

    return {"status": "done", "file": path}
