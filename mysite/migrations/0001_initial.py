# Generated by Django 2.2.12 on 2020-04-06 07:48

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Words',
            fields=[
                ('word', models.CharField(max_length=50, primary_key=True, serialize=False)),
                ('counter', models.BigIntegerField(default=0)),
            ],
        ),
    ]
