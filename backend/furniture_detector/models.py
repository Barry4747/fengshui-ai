from django.db import models


class Picture(models.Model):
    id = models.CharField(max_length=100, primary_key=True)
    image_path = models.CharField(max_length=255)

    detected_data = models.JSONField(blank=True, null=True)

    def __str__(self):
        return f"Picture {self.id}"


class Task(models.Model):
    id = models.CharField(max_length=100, primary_key=True)
    session_id = models.CharField(max_length=100)

    def __str__(self):
        return f"Task {self.id} (session {self.session_id})"


class Furniture(models.Model):
    name = models.CharField(max_length=100)
    x = models.FloatField(blank=True, null=True)
    y = models.FloatField(blank=True, null=True)
    z = models.FloatField(blank=True, null=True)

    picture = models.ForeignKey(Picture, on_delete=models.CASCADE, related_name="furnitures")
    task = models.ForeignKey(Task, on_delete=models.CASCADE, related_name="furnitures")

    def __str__(self):
        return f"{self.name} ({self.id})"
