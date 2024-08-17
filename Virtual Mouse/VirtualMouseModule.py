import cv2
from Hand_Gesture import Hand_tracking_module as htm
import time
import mouse
import pyautogui
import numpy as np
from collections import deque

wScr, hScr = pyautogui.size()
wCam, hCam = 640, 480

frameR = 100

buffer_size = 5
smoothing = 3

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
cTime = 0

pLocX, pLocY = 0, 0
cLocX, cLocY = 0, 0

detector = htm.handDectector(detectionCon=0.75)
position_buffer = deque(maxlen=buffer_size)


# smoothing using buffer
def smooth_values(current_position):
    position_buffer.append(current_position)
    avg_position = np.array(position_buffer).mean(axis=0)
    return int(avg_position[0]), int(avg_position[1])


while True:
    if cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            print("Error: Could not read frame.")
            break
        frame, numHands = detector.findHands(frame)
        lmlist = detector.findPosition(frame, draw=True)

        if lmlist:
            fingers = detector.fingersUp()
            # cv2.putText(frame, f'Fingers: {x1}, {y1}', (50, 200), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

            cv2.rectangle(frame, (frameR, frameR), (wCam - frameR - 50, hCam - frameR - 50), (255, 0, 255), 2, cv2.LINE_AA)
            # scrolling mode
            x, y = lmlist[8][1], lmlist[8][2]
            # smooth_x, smooth_y = smooth_values((x, y))
            length1, cords1 = detector.findDistance(4, 8, frame)
            length2, cords2 = detector.findDistance(8, 12, frame)
            if fingers[1:] == [ 1, 0, 0, 0]:
                screen_x = np.interp(x, (frameR, wCam - frameR - 50), (0, wScr))
                screen_y = np.interp(y, (frameR, hCam - frameR - 50), (0, hScr))

                # smoothing using exponential moving average
                cLocX = pLocX + (screen_x - pLocX) / smoothing
                cLocY = pLocY + (screen_y - pLocY) / smoothing

                # mouse.move(screen_x, screen_y)
                pyautogui.moveTo(cLocX, cLocY)
                cv2.circle(frame, (x, y), 8, (0, 255, 0), -1, cv2.FILLED)
                pLocX, pLocY = cLocX, cLocY

                # cv2.putText(frame, f'Mouse position: {mouse.get_position()}', (50, 200), cv2.FONT_HERSHEY_PLAIN,
                # 1.5, (255, 0, 0), 2) cv2.putText(frame, f'Mouse position as per finger: {x3,y3}', (50, 250),
                # cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)

            # left-clicking mode
            # if fingers == [1, 1, 0, 0, 0]:
            if length1 <= 20:
                # cv2.circle(frame, (lmlist[4][1], lmlist[4][2]), 15, (0, 255, 0), cv2.FILLED)
                mouse.click(button='left')
            #     pyautogui.mouseDown(button='left')
            #     # cv2.waitKey(100)
            # else:
            #     pyautogui.mouseUp(button='left')

            # right-clicking mode
            if fingers == [0, 1, 1, 0, 0]:
                if length2 <= 20:
                    # mouse.press(button='right')
                    pyautogui.click(button='right')

            if fingers == [1, 0, 0, 0, 1]:
                break

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame, f'FPS: {int(fps)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
        cv2.imshow("Camera Preview", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
