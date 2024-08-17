# import cv2
# import mediapipe as mp
# import time
#
# cap = cv2.VideoCapture(0)
# cap.set(3,1535)
# cap.set(4,863)
#
# mpHands = mp.solutions.hands
# hands = mpHands.Hands()
# mpDraw = mp.solutions.drawing_utils
#
# pTime = 0
# cTime = 0
#
# while True:
#     success, img = cap.read()
#     img = cv2.flip(img, 1)
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = hands.process(imgRGB)
#
#     if results.multi_hand_landmarks:
#         print(len(results.multi_hand_landmarks))
#         for handIndex, handLms in enumerate(results.multi_hand_landmarks):
#             # print(f'Hand {handIndex + 1}:')
#             for id, lm in enumerate(handLms.landmark):
#                 h, w, c = img.shape
#                 cx, cy = int(lm.x * w), int(lm.y * h)
#                 # print(f'Landmark {id}: ({cx}, {cy})')
#
#                 # Optionally draw circles on specific landmarks
#                 if id == 8:
#                     cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
#                     cv2.putText(img, f'Hand no:{handIndex+1}, ({cx}, {cy})', (cx, cy), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 1)
#
#             mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
#
#     cTime = time.time()
#     fps = 1 / (cTime - pTime)
#     pTime = cTime
#     cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
#
#     cv2.imshow("Image", img)
#     if cv2.waitKey(1) & 0xFF == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handIndex, handLms in enumerate(results.multi_hand_landmarks):
            # Get bounding box coordinates
            h, w, c = img.shape
            xmin, ymin = w, h
            xmax, ymax = 0, 0

            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                xmin, ymin = min(xmin, cx), min(ymin, cy)
                xmax, ymax = max(xmax, cx), max(ymax, cy)

            # Draw rectangle around the hand
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
