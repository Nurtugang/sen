from datetime import datetime
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
from django.db.models import Max, F
from django.core.serializers.json import DjangoJSONEncoder

from ultralytics import YOLO
import face_recognition
from yolox.tracker.byte_tracker import BYTETracker
from supervision.detection.core import BoxAnnotator

from .nurutils.detections import *
import cv2
import numpy as np
import ast
import json
from requests.auth import HTTPDigestAuth
import requests
from .models import Stata

# Пример функции, которая запускает обнаружение лиц
def run_face_detection():
    import os

    current_dir = os.getcwd()
    print("Текущий рабочий каталог:", current_dir)

    known_face_encodings = []
    known_face_names = ["ersa", "serik", "nur"]  # Adjust names accordingly
    encoding_files = ['/home/nurtugang17/DigitalFarabi/DigitalFarabi/core/encodings133/ersa.txt', '/home/nurtugang17/DigitalFarabi/DigitalFarabi/core/encodings133/serik.txt', '/home/nurtugang17/DigitalFarabi/DigitalFarabi/core/encodings133/nur.txt']  # Update paths

    for file in encoding_files:
        with open(file, 'r') as f:
            encoding_list = ast.literal_eval(f.read())
            known_face_encodings.append(encoding_list)


    # Инициализация YOLO
    MODEL = 'models/yolov8n.pt'
    model = YOLO(MODEL)
    model.predict(source="0", show=False, stream=True, classes=0)
    model.fuse()

    # Инициализация трекера и аннотатора
    byte_tracker = BYTETracker(BYTETrackerArgs())
    byte_tracker2 = BYTETracker(BYTETrackerArgs())

    box_annotator = BoxAnnotator(thickness=1, text_scale=0.5, text_thickness=1)


    urls = [
        'rtsp://admin:admin123@192.168.1.69:554/cam/realmonitor?channel=1&subtype=1',
        'rtsp://admin:admin123@192.168.1.79:554/cam/realmonitor?channel=1&subtype=1'
    ]

    geolocations = [
        (43.224457810767866, 76.92456475041607),
        (43.22451742183684, 76.92445477983628)
    ]

    window_names = ['Camera1', 'Camera2']


    video1 = cv2.VideoCapture(urls[0])
    video2 = cv2.VideoCapture(urls[1])
        
    while True:
        ret, frame = video1.read()
        ret2, frame2 = video2.read()

        results = model(frame)
        results2 = model(frame2)

        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int)
        )
        detections2 = Detections(
            xyxy=results2[0].boxes.xyxy.cpu().numpy(),
            confidence=results2[0].boxes.conf.cpu().numpy(),
            class_id=results2[0].boxes.cls.cpu().numpy().astype(int)
        )


        detections = detections[detections.class_id == 0]  # Фильтрация по классу (например, люди)
        detections2 = detections2[detections2.class_id == 0]  # Фильтрация по классу (например, люди)

        # Трекинг объектов
        tracks = byte_tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=frame.shape,
            img_size=frame.shape
        )
        tracks2 = byte_tracker2.update(
            output_results=detections2boxes(detections=detections2),
            img_info=frame2.shape,
            img_size=frame2.shape
        )

        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)  
        tracker_id2 = match_detections_with_tracks(detections=detections2, tracks=tracks2)     

        detections.tracker_id = np.array(tracker_id)
        detections2.tracker_id = np.array(tracker_id2) 

        mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool) 
        mask2 = np.array([tracker_id2 is not None for tracker_id2 in detections2.tracker_id], dtype=bool) 

        detections.filter(mask=mask, inplace=True) 
        detections2.filter(mask=mask2, inplace=True) 

        labels = [ 
            f"#{tracker_id} 'Person' {confidence:0.2f}" 
            for _, confidence, class_id, tracker_id 
            in detections 
        ]

        labels2 = [ 
            f"#{tracker_id2} 'Person' {confidence2:0.2f}" 
            for _2, confidence2, class_id2, tracker_id2 
            in detections2 
        ]

        # Аннотация кадра
        frame = box_annotator.annotate(scene=frame, detections=detections)

        frame2 = box_annotator.annotate(scene=frame2, detections=detections2)

        # Распознавание лиц
        for i in detections.xyxy:
            x1, y1, x2, y2 = map(int, i[:4])
            cropped = frame[y1:y2, x1:x2]
            scale_percent = 3.20

            new_width = int(cropped.shape[1] / scale_percent)  # Reverse the scaling
            new_height = int(cropped.shape[0] / scale_percent)  # Reverse the scaling
            dim = (new_width, new_height)
            frame_face = cv2.resize(cropped, dim, interpolation=cv2.INTER_AREA) 
            face_locations = face_recognition.face_locations(frame_face)
            face_encodings = face_recognition.face_encodings(frame_face, face_locations)

            for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    # Создание скриншота области с лицом
                    _, buffer = cv2.imencode('.jpg', frame)
                    face_image_file = ContentFile(buffer.tobytes())

                    #ВРЕМЯ
                    result_string = requests.get('http://192.168.1.69/cgi-bin/global.cgi?action=getCurrentTime', stream=True, auth=HTTPDigestAuth('admin', 'admin123')).text
                    date_time_str = result_string.split('=')[1].strip()
                    date_time_obj = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
                    # Создание новой записи в базе данных
                    stata = Stata(detection_time=date_time_obj, face_name=name, camera_name=window_names[0], camera_url=urls[0], latitude=geolocations[0][0], longitude=geolocations[0][1])
                    stata.face_image.save(f"{name}_{date_time_str}.jpg", face_image_file)

                top = int(top * scale_percent) + y1
                right = int(right * scale_percent) + x1
                bottom = int(bottom * scale_percent) + y1
                left = int(left * scale_percent) + x1

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)
        
        for i in detections2.xyxy:
            x1, y1, x2, y2 = map(int, i[:4])
            cropped2 = frame2[y1:y2, x1:x2]
            scale_percent = 3.20

            new_width2 = int(cropped2.shape[1] / scale_percent)  # Reverse the scaling
            new_height2 = int(cropped2.shape[0] / scale_percent)  # Reverse the scaling
            dim2 = (new_width2, new_height2)
            frame_face2 = cv2.resize(cropped2, dim2, interpolation=cv2.INTER_AREA) 
            face_locations2 = face_recognition.face_locations(frame_face2)
            face_encodings2 = face_recognition.face_encodings(frame_face2, face_locations2)

            for face_encoding, (top, right, bottom, left) in zip(face_encodings2, face_locations2):
                matches2 = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                face_distances2 = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index2 = np.argmin(face_distances2)
                if matches2[best_match_index2]:
                    name = known_face_names[best_match_index2]
                    # Создание скриншота области с лицом
                    _, buffer = cv2.imencode('.jpg', frame2)
                    face_image_file = ContentFile(buffer.tobytes())

                    #ВРЕМЯ
                    result_string = requests.get('http://192.168.1.79/cgi-bin/global.cgi?action=getCurrentTime', stream=True, auth=HTTPDigestAuth('admin', 'admin123')).text
                    date_time_str = result_string.split('=')[1].strip()
                    date_time_obj = datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')
                    # Создание новой записи в базе данных
                    stata = Stata(detection_time=date_time_obj, face_name=name, camera_name=window_names[1], camera_url=urls[1], latitude=geolocations[1][0], longitude=geolocations[1][1])
                    stata.face_image.save(f"{name}_{date_time_str}.jpg", face_image_file)
                top = int(top * scale_percent) + y1
                right = int(right * scale_percent) + x1
                bottom = int(bottom * scale_percent) + y1
                left = int(left * scale_percent) + x1

                cv2.rectangle(frame2, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame2, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 1)


        
        cv2.imshow(window_names[0], frame)
        cv2.imshow(window_names[1], frame2)


        if cv2.waitKey(1) & 0xFF == ord('\r'):
            video1.release()
            video2.release()
            cv2.destroyAllWindows()


    # Release video sources
    # video1.release()
    # video2.release()


@csrf_exempt  # Отключаем CSRF для простоты, для продакшена лучше использовать CSRF-токены
def start_face_detection(request):
    if request.method == "POST":
        # Запуск обнаружения лиц
        run_face_detection()
        return JsonResponse({"status": "started"})
    else:
        return JsonResponse({"error": "This endpoint accepts only POST requests."})


def index(request):
    context = {

    }
    return render(request, 'index.html', context)


def dashboard(request):
    statas = Stata.objects.all().order_by('-detection_time')
    context = {
        'statas': statas,
    }
    return render(request, 'dashboard.html', context)


def marshroute(request):
    persons = Stata.objects.all().values_list('face_name', flat=True).distinct()
    face_images = []
    for person in persons:
        face_images.append(Stata.objects.filter(face_name=person).last().face_image)
    content = zip(persons, face_images)
    context = {
        'content':content,
        'persons': persons,
        'face_images': face_images
    }
    return render(request, 'marshroute.html', context)

def gmap(request):
    name = request.GET.get('name')

    latest_observations = (
    Stata.objects
    .values('camera_name')  # Группировка по названию камеры
    .annotate(
        latest_date=Max('detection_time'),  # Получение последней даты для каждой группы
        latitude=F('latitude'),  # Сохранение значения широты для каждой записи
        longitude=F('longitude')  # Сохранение значения долготы для каждой записи
        )
    )
    observations_json = json.dumps(list(latest_observations), cls=DjangoJSONEncoder)
    context = {
        'observations': observations_json
    }
    return render(request, 'gmap.html', context)

def videoresults(request):
    context = {

    }
    return render(request, 'videoresults.html', context)
