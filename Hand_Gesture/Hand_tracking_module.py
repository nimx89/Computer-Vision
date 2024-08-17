import cv2
import mediapipe as mp
import time
import math
import numpy as np


class handDectector():

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.lmList = []
        self.tipIDs = [4, 8, 12, 16, 20]
        self.numHands = 0

    def findHands(self, img, draw=False):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # MediaPipe hands only accepts RGB images
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            self.numHands = len(self.results.multi_hand_landmarks)
            for handLms in self.results.multi_hand_landmarks:
                if draw:

                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)  # Draw the landmarks(by handLms) and the connections(by mpHands.HAND_CONNECTIONS)

        return img, self.numHands



    def findPosition(self, img, handNo=0, draw=False):

        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, c = img.shape  # Get the height, width and channel of the image
            xmin, ymin = w, h
            xmax, ymax = 0, 0
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)  # Convert the landmarks from normalized values to pixel values
                xmin, ymin = min(xmin, cx), min(ymin, cy)
                xmax, ymax = max(xmax, cx), max(ymax, cy)
                # print(id, cx, cy)
                self.lmList.append([id, cx, cy])
                # how to get index of each landmark
            if draw:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(img, f'Hand no:{handNo + 1}', (xmin, ymin - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

        return self.lmList

    def fingersUp(self, lmList_local = None):
        if lmList_local is None:
            lmList_local = self.lmList

        if len(lmList_local) != 0:
            fingers = []

            if lmList_local[4][1] < lmList_local[0][1]:
                # only for left hand
                if lmList_local[4][1] <= lmList_local[2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                # only for right hand
                if lmList_local[4][1] >= lmList_local[2][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            for ids in range(1, 5):
                if lmList_local[self.tipIDs[ids]][2] < lmList_local[self.tipIDs[ids] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            return fingers
        return []

    def findDistance(self, p1, p2, img, draw=False, r=15, t=3 , lmList_local = None):
        """
        p1: landmark1 index, p2: landmark2 index, img: image, draw: draw the line, r: radius of the circle, t: thickness of the line
        return: length of the line, img, [x1, y1, x2, y2, cx, cy]
        """
        if lmList_local is None:
            lmList_local = self.lmList
        x1, y1 = lmList_local[p1][1:]
        x2, y2 = lmList_local[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (255, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)
        return length, [cx, cy]


#Dummy mainfunction can be used in any other file

def main():
    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture(0)
    detector = handDectector()
    while True:
        success, img = cap.read()
        img, numHands = detector.findHands(img,draw =False)
        lmlist = detector.findPosition(img, draw=True)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255),
                    3)  # display the fps on the image

        cv2.imshow("Image", img)
        cv2.waitKey(1)

        if cv2.waitKey(1) & 0xFF == 27:
            break

if __name__ == "__main__":
    main()
