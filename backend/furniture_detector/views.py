from django.shortcuts import get_object_or_404
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import *
from .models import Task, Picture
import uuid
from django.conf import settings
import os
from .tasks import process_image

# -----------------------
# Upload image
# -----------------------
@api_view(['POST'])
def upload_image(request):
    serializer = UploadSerializer(data=request.data)

    session_id = request.data.get('session_id')
    task_id = request.data.get('task_id')

    if not session_id or not task_id:
        return Response({"error": "Session id and task id must be provided"}, status=400)

    if serializer.is_valid():
        image = serializer.validated_data['file']

        task, created = Task.objects.get_or_create(
            id=task_id, defaults={"session_id": session_id}
        )
        picture_id = str(uuid.uuid4())
        path = os.path.join(settings.MEDIA_ROOT, f'{picture_id}.jpg')
        with open(path, 'wb+') as f:
            for chunk in image.chunks():
                f.write(chunk)

        picture = Picture.objects.create(id=picture_id, image_path=path)

        process_image.delay(task.id, picture.id, path)

        return Response({"task_id": task.id, "picture_id": picture.id})

    return Response(serializer.errors, status=400)



# -----------------------
# Calculate dimensions
# -----------------------
@api_view(['POST'])
def calculate_dimensions(request):
    """
    Body JSON:
    {
        "reference_length": 50,   # cm
        "detected_length": 200    # px
    }
    """
    ref_len = float(request.data.get("reference_length", 0))
    detected_len = float(request.data.get("detected_length", 0))

    if ref_len <= 0 or detected_len <= 0:
        return Response({"error": "Invalid values"}, status=400)

    scale = ref_len / detected_len
    return Response({"scale_factor": scale})
