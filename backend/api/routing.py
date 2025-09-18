from django.urls import path, re_path
from furniture_detector.routing import websocket_urlpatterns as fd_patterns

websocket_urlpatterns = [
    *fd_patterns,
]

