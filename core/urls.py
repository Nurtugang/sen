from django.urls import path 
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('start-face-detection/', views.start_face_detection, name='start_face_detection'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('marshroute/', views.marshroute, name='marshroute'),
    path('videoresults/', views.videoresults, name='videoresults'),
    path('gmap/', views.gmap, name='gmap'),

]
