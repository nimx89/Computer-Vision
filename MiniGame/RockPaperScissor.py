import cv2
import numpy as np
from Hand_Gesture import Hand_tracking_module as htm
import time
import keyboard

cap = cv2.VideoCapture(0)
wCam = int(cap.get(3))
hCam = int(cap.get(4))

detector = htm.handDectector(detectionCon=0.75)
frameR = 100

pTime = 0
cTime = 0


def get_random_number():
    import random
    return random.randint(0, 2)


def countdown(seconds):
    for i in range(seconds, 0, -1):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, str(i), (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 255, 0), 4, cv2.LINE_AA)
        cv2.imshow('Rock Paper Scissors', frame)
        time.sleep(1)

    ret, frame = cap.read()
    if ret:
        cv2.putText(frame, 'Go!', (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 255, 0), 4, cv2.LINE_AA)
        cv2.imshow('Rock Paper Scissors', frame)
        time.sleep(1)


playing_mode = False

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    cropped_img = img[frameR: hCam - frameR, frameR: wCam - frameR]
    cropped_img, numHands = detector.findHands(cropped_img, draw=True)
    lmlist = detector.findPosition(cropped_img, draw=True)
    user_choice = -1
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2, cv2.LINE_AA)
    if not playing_mode:
        cv2.putText(img, "Press Space to play", (100, 95), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1, cv2.LINE_AA)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            playing_mode = True
            # countdown(3)

    if playing_mode:
        cv2.putText(img, "Play Rock, Paper, or Scissor", (100, 80), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 1)

        if lmlist:
            fingers = detector.fingersUp()
            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2, cv2.LINE_AA)

            if fingers == [0, 0, 0, 0, 0]:
                user_choice = 0
            elif fingers == [1] * 5:
                user_choice = 1
            elif fingers == [0, 1, 1, 0, 0]:
                user_choice = 1

            if user_choice != -1:

            computer_choice = get_random_number()
            if computer_choice == 0:
                cv2.putText(img, "Computer: Rock", (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            elif computer_choice == 1:
                cv2.putText(img, "Computer: Paper", (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            else:
                cv2.putText(img, "Computer: Scissor", (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

            if user_choice == computer_choice:
                cv2.putText(img, "You Win", (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                cv2.waitKey()

            else:
                cv2.putText(img, "You Lose", (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)

    cv2.imshow("Playing Mode", img)


    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
