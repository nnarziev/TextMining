from django.db import models


# Create your models here.
class Words(models.Model):
    text = models.TextField()
    year = models.IntegerField(default=0)
    count = models.BigIntegerField(default=0)

    class Meta:
        unique_together = (('text', 'year'),)


class Collocations(models.Model):
    text = models.TextField()
    year = models.IntegerField(default=0)
    count = models.BigIntegerField(default=0)

    class Meta:
        unique_together = (('text', 'year'),)


class Embeddings(models.Model):
    text = models.TextField(primary_key=True)
    embedding = models.BinaryField()
