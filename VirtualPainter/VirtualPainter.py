import cv2
import mediapipe
import time
import numpy as np
import os
from Hand_Gesture import Hand_tracking_module as htm

imgCanvas = np.zeros((720, 1280, 3), np.uint8)
drawingColor = (255, 0, 255)
pX, pY = 0, 0
wCam, hCam = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
detector = htm.handDectector(detectionCon=0.75)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    img ,numHand= detector.findHands(img)
    lmList = detector.findPosition(img, draw=True)

    if len(lmList) != 0:
        fingers = detector.fingersUp()
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # for scrolling mode
        if fingers[1] and fingers[2]:
            if y1 < 125:
                if 250 < x1 < 450:
                    drawingColor = (255, 0, 255)
                elif 550 < x1 < 750:
                    drawingColor = (0, 255, 0)
                elif 800 < x1 < 950:
                    drawingColor = (0, 0, 255)
                elif 1050 < x1 < 1200:
                    drawingColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawingColor, cv2.FILLED)

        # for drawing mode
        if fingers[1] and not fingers[2]:
            cv2.circle(img, (x1, y1), 15, drawingColor, cv2.FILLED)
            if pX == 0 and pY == 0:
                pX, pY = x1, y1

            cv2.line(img, (pX, pY), (x1, y1), drawingColor, 15)
            cv2.line(imgCanvas, (pX, pY), (x1, y1), drawingColor, 15)

            pX, pY = x1, y1

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)

    img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.merge((img, imgCanvas))
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
