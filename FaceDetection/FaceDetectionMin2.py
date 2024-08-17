import cv2
import mediapipe as mp
import time

#Just the same thing as the previous file, but with lil different type of code

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cTime = 0
pTime = 0

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    ret, frame = cap.read()

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow('frame', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
