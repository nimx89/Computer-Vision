import cv2
import time
import os
from Hand_Gesture import Hand_tracking_module as htm

wCam, hCam = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

fingerPath = 'FingerImages'
myList = os.listdir(fingerPath)
pTIme =0
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{fingerPath}/{imPath}')
    overlayList.append(image)


detector = htm.handDectector(detectionCon=0.75)
tipIDs = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    img = cv2.flip(img, 1)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        print(lmList[8])
        fingers = []

        if lmList[4][1] < lmList[0][1]:
            # only for left hand
            if lmList[4][1] < lmList[1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            # only for right hand
            if lmList[4][1] > lmList[2][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        for id in range(1,5):
            if lmList[tipIDs[id]][2] < lmList[tipIDs[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers = fingers.count(1)
        # print(fingers)

        # h,w,c = overlayList[totalFingers  ].shape
        # img[0:h , 0:w] = overlayList[totalFingers ]
        if totalFingers == 5 :
            cv2.putText(img, "High five", (200,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)

        if fingers == [0,1,1,0,0]:
            cv2.putText(img, "Victory sign", (200,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)


    cTime = time.time()
    fps = 1 / (cTime - pTIme)
    pTIme = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Image', img)
    cv2.waitKey(1)

