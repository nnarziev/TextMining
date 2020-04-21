from django.db import models


# Create your models here.
class Words(models.Model):
    text = models.CharField(max_length=50)
    year = models.IntegerField(default=0)
    count = models.BigIntegerField(default=0)

    class Meta:
        unique_together = (('text', 'year'),)


class Bigrams(models.Model):
    text = models.TextField()
    year = models.IntegerField(default=0)
    count = models.BigIntegerField(default=0)

    class Meta:
        unique_together = (('text', 'year'),)


class Trigrams(models.Model):
    text = models.TextField()
    year = models.IntegerField(default=0)
    count = models.BigIntegerField(default=0)

    class Meta:
        unique_together = (('text', 'year'),)
