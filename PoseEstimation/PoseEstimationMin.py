import cv2
import mediapipe as mp
import time
from Hand_Gesture import Hand_tracking_module as htm

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
detector = htm.handDectector()
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,1000)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT,10)

pTime = 0
cTime = 0
while True:

    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = detector.findHands(img)

    results = pose.process(imgRGB)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            print(lm,id)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(5)# 1 millisecond delay between frames

