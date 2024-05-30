from django.db import models
import os

# Create your models here.


class Actions(models.Model):
    images = models.ImageField(upload_to='webapp/static/images')

    def filename(self):
        return os.path.basename(self.images.name)
