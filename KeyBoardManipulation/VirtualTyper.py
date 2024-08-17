import cv2
import keyboard
from Hand_Gesture import Hand_tracking_module as htm

wCam, hCam = 1535, 863

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDectector(detectionCon=0.75)
freq = False
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        print("Error: Could not read frame.")
        break
    frame, numHands = detector.findHands(frame, draw=False)
    lmList = detector.findPosition(frame, handNo=0, draw=True)
    lmList_curr = None
    if lmList:
        fingers = detector.fingersUp(lmList_curr)
        if numHands > 1:
            lmList2 = detector.findPosition(frame, handNo=1, draw=True)
            fingers2 = detector.fingersUp(lmList2)
            if fingers == [1, 0, 0, 0, 0] and fingers2 == [1, 0, 0, 0, 0]:
                break
            lmList_curr = lmList2

        # Using fingers up and down to control keyboard
        # # use thumb to press and hold alt
        # if fingers[0] == 0:
        #     keyboard.press('alt')
        #     cv2.waitKey(100)
        # else:
        #     keyboard.release('alt')
        #
        # # use ring finger to press tab
        # if fingers[3] == 0:
        #     keyboard.press('tab')
        #     cv2.waitKey(100)
        # else:
        #     keyboard.release('tab')


        length = detector.findDistance(4, 8, frame, draw=False, lmList_local=lmList_curr)[0]
        length2 = detector.findDistance(8, 12, frame, draw=False, lmList_local= lmList_curr)[0]

        # cv2.putText(frame, f'Length: {int(length)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        if length < 30:
            if not freq:
                # print('Alt+Tab')
                keyboard.press('alt+tab')
                cv2.waitKey(100)
                keyboard.release('tab')
                freq = True

            if length2 < 30:
                keyboard.press('tab')
                cv2.waitKey(100)
        else:
            keyboard.release('alt')
            keyboard.release('tab')
            freq = False

    cv2.imshow('Camera Preview', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()