import cv2
import numpy as np


# Функция для обнаружения лиц
def detect_faces(image):
    # Загружаем каскад Haar-классификатора
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Применяем каскад к изображению
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    # Применяем обученную модель для определения лица
    is_allowed = True  # check face for allowed
    # Рисуем рамки вокруг лиц в зависимости от классификации
    for (x, y, w, h) in faces:
        if is_allowed:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # green
        else:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # red
    return image


cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened():  # try to get the first frame
    r_val, frame = vc.read()
else:
    r_val = False

while r_val:
    cv2.imshow("preview", frame)
    r_val, frame = vc.read()
    key = cv2.waitKey(10)
    if key == 27:  # exit on ESC
        break

    detect_faces(frame)

vc.release()
cv2.destroyWindow("preview")
