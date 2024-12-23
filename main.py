import os
import cv2

import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from hand_landmarker import HandLandmarker

from enums import Mode


# Import header images
folder_path = "img"
list_of_imgs = os.listdir(folder_path)

overlay_list = []

for img_path in list_of_imgs:
    image = cv2.imread(f'{folder_path}/{img_path}')
    overlay_list.append(image)
print(len(overlay_list)) # safety measure

header = overlay_list[0]

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4, 720)

circle_coords = []

hand_landmarker = HandLandmarker()

canvas = np.zeros((720,1280,3), dtype=np.uint8)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image from webcam.")
        break


    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    detection_result = hand_landmarker.detector.detect(mp_image)

    # draws annotations for hands
    annotated_result = hand_landmarker.draw_landmarks_on_image(img, detection_result)

    annotated_result[0:100, 0:1280] = header

    cv2.imshow("Hand Landmarks", annotated_result)

    if hand_landmarker.activity_mode == Mode.DRAWING:
        cv2.circle(canvas, hand_landmarker.get_pixel_coordinates(8, 1280, 720), 5, (255, 0, 0))

    cv2.imshow("Canvas", canvas)

    hand_landmarker.check_mode()
    print(hand_landmarker.activity_mode)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
