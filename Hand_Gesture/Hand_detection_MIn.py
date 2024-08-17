import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1080)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # MediaPipe hands only accepts RGB images
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        # print(results.multi_hand_landmarks)
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h,w,c = img.shape # Get the height, width and channel of the image
                cx, cy = int(lm.x*w), int(lm.y*h) # Convert the landmarks from normalized values to pixel values
                # print(id, cx, cy)
                #how to get index of each landmark
                if id == 1:
                    # cv2.circle(img, (cx,cy), 25, (255,0,255), cv2.FILLED)
                    cv2.putText(img, f'{int(cx), int(cy)}', (cx,cy), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,255), 1)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # Draw the landmarks(by handLms) and the connections(by mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,0,255), 1) # display the fps on the image

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
