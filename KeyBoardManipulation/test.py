# Using Keyboard module in Python
import keyboard
import cv2
from Hand_Gesture import Hand_tracking_module as htm

wCam, hCam = 1535, 863
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDectector(detectionCon=0.75)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = detector.findHands(frame, draw=False)

