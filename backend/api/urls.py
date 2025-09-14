from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from .views import health

urlpatterns = [
    path('health', health, name='health_check'),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)