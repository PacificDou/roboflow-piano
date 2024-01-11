from roboflow import Roboflow
import os
import cv2
import time
import base64
from datetime import datetime
import requests
import numpy as np
import pygame
import os
import sys

IMG_PATH = 'image.jpg'
API_KEY = os.getenv('API_KEY')
DISTANCE_TO_OBJECT = 500  # mm
SOUND_FOLDER = "../sounds" #
GAZE_DETECTION_URL = "http://localhost:9001/gaze/gaze_detection"


def detect_gazes(frame):
    img_encode = cv2.imencode('.jpg', frame)[1]
    img_base64 = base64.b64encode(img_encode)
    resp = requests.post(GAZE_DETECTION_URL,
                         json={"api_key": API_KEY, "image": {"type": "base64", "value": img_base64.decode("utf-8")}})
    gazes = resp.json()[0]['predictions']
    return gazes


def draw_gaze(img, gaze):
    # draw face bounding box
    face = gaze['face']
    x_min = int(face['x'] - face['width'] / 2)
    x_max = int(face['x'] + face['width'] / 2)
    y_min = int(face['y'] - face['height'] / 2)
    y_max = int(face['y'] + face['height'] / 2)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

    # draw gaze arrow
    imgH, imgW = img.shape[:2]
    arrow_length = imgW / 2
    dx = arrow_length * np.sin(gaze['yaw']) * np.cos(gaze['pitch'])
    dy = -arrow_length * np.sin(gaze['pitch'])
    cv2.arrowedLine(img, (int(face['x']), int(face['y'])), (int(face['x'] + dx), int(face['y'] + dy)),
                    (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.18)

    # draw keypoints
    for keypoint in face['landmarks']:
        color, thickness, radius = (0, 255, 0), 2, 2
        x, y = int(keypoint['x']), int(keypoint['y'])
        cv2.circle(img, (x, y), thickness, color, radius)

    # draw label and score
    label = "yaw {:.2f}  pitch {:.2f}".format(gaze['yaw'] / np.pi * 180, gaze['pitch'] / np.pi * 180)
    cv2.putText(img, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    return img


if __name__ == "__main__":

    # main workflow
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            print("Error happened while reading image")
            break

        # gaze detection
        gazes = detect_gazes(frame)
        if len(gazes) == 0:
            print("No face detected...")
            continue

        # draw face & gaze
        gaze = gazes[0]
        draw_gaze(frame, gaze)
        
        # calculate unit length per pixel (mm), assume human face height is 250mm
        face = gaze['face']
        length_per_pixel = 250 / face['height']

        # calculate & draw gaze point
        dx = DISTANCE_TO_OBJECT * np.tan(gaze['yaw']) / length_per_pixel
        dx = dx if not np.isnan(dx) else frame.shape[1] * 10
        dy = -DISTANCE_TO_OBJECT * np.arccos(gaze['yaw']) * np.tan(gaze['pitch']) / length_per_pixel
        dy = dy if not np.isnan(dy) else frame.shape[0] * 10
        gaze_point = int(frame.shape[1] / 2 + dx), int(frame.shape[0] / 2 + dy)
        cv2.circle(frame, gaze_point, 20, (0, 0, 255), -1)

        # show latest views
        cv2.imshow("gaze", frame)

        if cv2.waitKey(500) == 27:
            break
