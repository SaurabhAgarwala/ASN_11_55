from django.db import models

# Create your models here.
class Violation(models.Model):
    multiperson = models.IntegerField(default=0, blank=False)
    fullscreen = models.IntegerField(default=0, blank=False)
    screenfocus = models.IntegerField(default=0, blank=False)
    audio = models.IntegerField(default=0, blank=False)
    facedir = models.IntegerField(default=0, blank=False)
    facesim = models.IntegerField(default=0, blank=False)
    othergadgets = models.IntegerField(default=0, blank=False)
