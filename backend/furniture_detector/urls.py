from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_image, name='upload'),
    path('dimensions/', views.calculate_dimensions, name='dimensions'),
]
