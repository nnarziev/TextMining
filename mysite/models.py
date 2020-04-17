from django.db import models

# Create your models here.
class Words(models.Model):
    word = models.CharField(primary_key=True, max_length=50)
    counter = models.BigIntegerField(default=0)