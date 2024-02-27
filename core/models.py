from django.db import models


class Stata(models.Model):
    detection_time = models.DateTimeField(auto_now_add=True)
    face_image = models.ImageField(upload_to='face_detections/', verbose_name='Скриншот')
    face_name = models.CharField(max_length=100, verbose_name='Имя')
    camera_name = models.CharField(max_length=100, verbose_name='Камера')
    camera_url = models.CharField(max_length=255, verbose_name='RTSP-ссылка камеры')
    latitude = models.FloatField()
    longitude = models.FloatField()

    def __str__(self):
        return f"{self.face_name} обнаруженный в {self.detection_time}"


