from django.db import models


# Create your models here.
class Words(models.Model):
    text = models.CharField(max_length=50)
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
    text = models.CharField(max_length=50, primary_key=True)
    embedding = models.BinaryField()


class WordsSimilarity(models.Model):
    text = models.CharField(max_length=50, primary_key=True)
    embedding = models.BinaryField()


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
