from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .serializers import *
import uuid
from django.conf import settings
import os
from .tasks import process_image

@api_view(['POST'])
def upload_image(request):
    serializer = UploadSerializer(data=request.data)
    if serializer.is_valid():
        image = serializer.validated_data['file']
        task_id = str(uuid.uuid4())
        path = os.path.join(settings.MEDIA_ROOT, f'{task_id}.jpg')
        with open(path, 'wb+') as f:
            for chunk in image.chunks():
                f.write(chunk)

        process_image.delay(task_id, path)

        return Response({"task_id": task_id})
    return Response(serializer.errors, status=400)

from celery.result import AsyncResult
from core.celery import app

@api_view(['GET'])
def check_status(request, task_id):
    result = AsyncResult(task_id, app=app)
    return Response({
        "task_id": task_id,
        "status": result.status,
        "result": result.result if result.ready() else None
    })


@api_view(['POST'])
def calculate_dimensions(request):
    """
    Przykładowy body:
    {
        "reference_length": 50,   # cm, np. długość znanej referencji
        "detected_length": 200    # px w zdjęciu
    }
    """
    ref_len = float(request.data.get("reference_length", 0))
    detected_len = float(request.data.get("detected_length", 0))

    if ref_len <= 0 or detected_len <= 0:
        return Response({"error": "Invalid values"}, status=400)

    scale = ref_len / detected_len
    return Response({"scale_factor": scale})
