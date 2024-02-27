# Generated by Django 4.2.6 on 2024-02-22 23:04

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Stata',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('detection_time', models.DateTimeField(auto_now_add=True)),
                ('face_image', models.ImageField(upload_to='face_detections/')),
                ('face_name', models.CharField(max_length=100)),
                ('camera_name', models.CharField(max_length=100)),
                ('camera_url', models.CharField(max_length=255)),
            ],
        ),
    ]
