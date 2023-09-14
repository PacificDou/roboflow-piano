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
DISTANCE_TO_OBJECT = 1000  # mm
SOUND_FOLDER = "../sounds" #
GAZE_DETECTION_URL = "http://127.0.0.1:9001/gaze/gaze_detection"


def load_products():
    # product info  calories--kcal, sugar--g
    products = {}

    products['7up_320ml'] = {'calories': 100, 'sugar': 25, 'sound': 'c4.wav'}
    products['7up_390ml'] = {'calories': 120, 'sugar': 31, 'sound': 'd4.wav'}
    products['7up_chat_xo_320ml'] = {'calories': 0, 'sugar': 0, 'sound': 'e4.wav'}

    products['aquafina_soda'] = {'calories': 0, 'sugar': 0, 'sound': 'f4.wav'}

    products['cocacola_390ml'] = {'calories': 172, 'sugar': 41, 'sound': 'g4.wav'}
    products['cocacola_600ml'] = {'calories': 258, 'sugar': 64, 'sound': 'a4.wav'}
    products['cocacola_zero_600ml'] = {'calories': 2, 'sugar': 0, 'sound': 'b4.wav'}

    products['lipton_450ml'] = {'calories': 5, 'sugar': 25, 'sound': 'c4.wav'}

    products['mirinda_cam_320ml'] = {'calories': 174, 'sugar': 47, 'sound': 'd4.wav'}
    products['mirinda_cam_390ml'] = {'calories': 212, 'sugar': 57, 'sound': 'e4.wav'}
    products['mirinda_sa_si_320ml'] = {'calories': 154, 'sugar': 41, 'sound': 'f4.wav'}
    products['mirinda_sa_si_390ml'] = {'calories': 187, 'sugar': 50, 'sound': 'g4.wav'}
    products['mirinda_soda_kem_320ml'] = {'calories': 148, 'sugar': 38, 'sound': 'a4.wav'}
    products['mirinda_soda_kem_390ml'] = {'calories': 180, 'sugar': 46, 'sound': 'b4.wav'}
    products['mirinda_viet_quat_320ml'] = {'calories': 218, 'sugar': 49, 'sound': 'c4.wav'}
    products['mirinda_viet_quat_390ml'] = {'calories': 265, 'sugar': 59, 'sound': 'd4.wav'}

    products['pepsi_320ml'] = {'calories': 90, 'sugar': 22, 'sound': 'e4.wav'}
    products['pepsi_390ml'] = {'calories': 109, 'sugar': 27, 'sound': 'f4.wav'}
    products['pepsi_zero_320ml'] = {'calories': 2, 'sugar': 0, 'sound': 'g4.wav'}
    products['pepsi_zero_390ml'] = {'calories': 2, 'sugar': 0, 'sound': 'a4.wav'}

    products['rockstar_250ml'] = {'calories': 130, 'sugar': 30, 'sound': 'b4.wav'}

    products['revive_original'] = {'calories': 40, 'sugar': 10, 'sound': 'c4.wav'}
    products['revive_salt_lemon'] = {'calories': 18, 'sugar': 5, 'sound': 'd4.wav'}

    products['sting_320ml'] = {'calories': 250, 'sugar': 62, 'sound': 'e4.wav'}
    products['sting_330ml'] = {'calories': 304, 'sugar': 76, 'sound': 'f4.wav'}
    products['sting_vang_320ml'] = {'calories': 134, 'sugar': 33, 'sound': 'g4.wav'}
    products['sting_vang_330ml'] = {'calories': 164, 'sugar': 41, 'sound': 'a4.wav'}

    products['tea_olong_320ml'] = {'calories': 83, 'sugar': 18, 'sound': 'b4.wav'}
    products['tea_olong_450ml'] = {'calories': 117, 'sugar': 25, 'sound': 'c4.wav'}
    products['tea_olong_chanh_450ml'] = {'calories': 5, 'sugar': 0, 'sound': 'd4.wav'}
    products['tea_olong_khong_duong_450ml'] = {'calories': 5, 'sugar': 0, 'sound': 'e4.wav'}

    products['twister_450ml'] = {'calories': 200, 'sugar': 116, 'sound': 'f4.wav'}
    products['twister_cam_c320ml'] = {'calories': 142, 'sugar': 82, 'sound': 'g4.wav'}

    return products


def detect_drinks():
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace().project("drink-detection")
    model = project.version(3).model
    resp = model.predict(IMG_PATH, confidence=40, overlap=30)
    resp.save("prediction.jpg")
    return resp.json()['predictions']


def draw_drink_bbox(img, drink):
    x_min = int(drink['x'] - drink['width'] / 2)
    x_max = int(drink['x'] + drink['width'] / 2)
    y_min = int(drink['y'] - drink['height'] / 2)
    y_max = int(drink['y'] + drink['height'] / 2)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
    cv2.putText(img, "{} ({:.2f})".format(drink['class'], drink['confidence']), (x_min, y_min - 10),
                cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
    return img


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
    dx = -arrow_length * np.sin(gaze['yaw']) * np.cos(gaze['pitch'])
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


def playsound(sound_file):
    if os.path.exists(SOUND_FOLDER + '/' + sound_file):
        pygame.mixer.music.load(SOUND_FOLDER + '/' + sound_file)
    else:
        pygame.mixer.music.load(SOUND_FOLDER + '/c4.wav')
    pygame.mixer.music.play()


def highlight_gazing_drinks(img, products, drinks, gaze_point):
    for drink in drinks:
        x_min = int(drink['x'] - drink['width'] / 2)
        x_max = int(drink['x'] + drink['width'] / 2)
        y_min = int(drink['y'] - drink['height'] / 2)
        y_max = int(drink['y'] + drink['height'] / 2)
        if not (x_min <= gaze_point[0] <= x_max and y_min <= gaze_point[1] <= y_max):
            continue

        cls_name = drink['class']
        if cls_name not in products:
            print("Cannot find product {}".format(cls_name))
            continue

        # show drink nutrition facts
        product = products[cls_name]
        cv2.putText(img, "Name: {}".format(cls_name), (10, 300), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 255), 2)
        cv2.putText(img, "Calories: {} kcal".format(product['calories']), (10, 400), cv2.FONT_HERSHEY_PLAIN, 5,
                    (0, 255, 255), 2)
        cv2.putText(img, "Sugar: {} g".format(product['sugar']), (10, 500), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 255), 2)

        # draw drink bounding box
        draw_drink_bbox(img, drink)

        playsound(product['sound'])
        print("gaze {}  img_size {}  object {}".format(gaze_point, img.shape, cls_name))

    return img


if __name__ == "__main__":
    pygame.mixer.init()

    # load product info   key: class name, value: product info
    products = load_products()

    # drink detection
    drinks = detect_drinks()
    if len(drinks) == 0:
        sys.exit("No drink detected...")

    # show drink bounding boxes
    img_drink = cv2.imread(IMG_PATH)
    for drink in drinks:
        draw_drink_bbox(img_drink, drink)
    cv2.imshow("drink", img_drink)
    # cv2.waitKey(0)

    # calculate unit length per pixel (mm), assume the drink width is 66.2mm
    length_per_pixel = 66.2 / np.mean([drink['width'] for drink in drinks])

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

        # reload drink image
        img_drink = cv2.imread(IMG_PATH)
        cv2.putText(img_drink, "{}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), (10, 100),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 2)

        # calculate & draw gaze point
        dx = DISTANCE_TO_OBJECT * np.tan(gaze['yaw']) / length_per_pixel
        dx = dx if not np.isnan(dx) else img_drink.shape[1] * 10
        dy = -DISTANCE_TO_OBJECT * np.arccos(gaze['yaw']) * np.tan(gaze['pitch']) / length_per_pixel
        dy = dy if not np.isnan(dy) else img_drink.shape[0] * 10
        gaze_point = int(img_drink.shape[1] / 2 + dx), int(img_drink.shape[0] / 2 + dy)
        cv2.circle(img_drink, gaze_point, 50, (0, 0, 255), -1)

        # highlight gazing object
        highlight_gazing_drinks(img_drink, products, drinks, gaze_point)

        # show latest views
        cv2.imshow("drink", img_drink)
        cv2.imshow("gaze", frame)

        if cv2.waitKey(500) == 27:
            break
