import cv2
import mediapipe as mp
import numpy as np
import time
from Hand_Gesture import Hand_tracking_module as htm
import math
import pycaw
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

########################################
wCam, hCam = 1280, 720
########################################

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4, hCam)

pTime = 0
detector = htm.handDectector(detectionCon= 0.75)

# #----------------------
# from __future__ import print_function
# from pycaw.pycaw import AudioUtilities, ISimpleAudioVolume
#
# #----------------------

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
# volume.SetMasterVolumeLevel(0, None)
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volbar = 400
volper = 0

while True:
    ret, frame = cap.read()
    frame = detector.findHands(frame )
    lmList = detector.findPosition(frame, draw = False)
    if lmList:
        # print(lmList[4], lmList[8])

        #for thumb-f0
        f0x , f0y = lmList[4][1], lmList[4][2]
        cv2.circle(frame, (f0x, f0y), 5, (255, 0, 100), cv2.FILLED)

        #for index finger f1
        f1x, f1y = lmList[8][1], lmList[8][2]
        cv2.circle(frame, (f1x, f1y), 5, (255, 0, 100), cv2.FILLED)

        cv2.line(frame,  (f0x, f0y), (f1x, f1y), (255, 0,100), thickness=1)

        cx, cy = (f0x+f1x)//2 , (f0y+f1y)//2
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        length = math.hypot(f1x - f0x, f1y - f0y)
        cv2.putText(frame, f'Length: {int(length)}', (f0x, f0y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        #Hand range 30 - 300
        #Volume Range -65 - 0

        vol = np.interp(length, [30, 200], [minVol, maxVol])
        # print(vol)
        volper = volume.GetMasterVolumeLevelScalar()*100
        volbar = np.interp(volper, [0, 100], [395, 155])
        # cv2.putText(frame, f'Volume: {(vol)}', (cx, cy), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        volume.SetMasterVolumeLevel(vol, None)


        if length < 30:
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(frame, (50, 150), (85, 400), ( 255,0, 0), 3)
    cv2.rectangle(frame, (55, int(volbar)), (80, 395), ( 0,255, 0), cv2.FILLED)
    cv2.putText(frame, f'{int(volper)} %', (90, 400), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    # cv2.putText(frame, f'Volume per: {volume.GetMasterVolumeLevelScalar()}', (100, 400), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    volper = np.interp(vol, [minVol, maxVol], [0, 100])


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)


    cv2.imshow('frame', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()