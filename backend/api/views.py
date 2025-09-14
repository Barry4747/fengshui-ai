from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
import requests

@api_view(['GET'])
def health(request):
    text = request.data.get('text')
    return Response(f'Copy that: {text}', status=200)
