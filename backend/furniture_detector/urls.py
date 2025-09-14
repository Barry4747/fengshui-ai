from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_image, name='upload'),
    path('status/<str:task_id>/', views.check_status, name='status'),
    path('dimensions/', views.calculate_dimensions, name='dimensions'),
]
