from django.contrib.auth.models import AbstractUser
from django.db import models

class CustomUser(AbstractUser):
    # add fields if needed

    def __str__(self):
        return self.username
