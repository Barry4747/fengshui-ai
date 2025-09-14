from django.db import models

class Furniture():
    name = models.CharField()
    x = models.FloatField(blank=True, null=True)
    y = models.FloatField(blank=True, null=True)
    z = models.FloatField(blank=True, null=True)

    session_id = models.CharField()


