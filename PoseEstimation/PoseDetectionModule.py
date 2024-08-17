import cv2
import mediapipe as mp
import time

class poseDectector():
    def __init__(self, mode=False, SmoothLandmarks=True, minDetectionCon=0.5, minTrackCon=0.5):

        self.mode = mode
        self.SmoothLandmarks = SmoothLandmarks
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, smooth_landmarks=self.SmoothLandmarks, min_detection_confidence=self.minDetectionCon, min_tracking_confidence=self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # MediaPipe hands only accepts RGB images
        self.results = self.pose.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.pose_landmarks:
            # for poseLms in self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)  # Draw the landmarks(by handLms) and the connections(by mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img,  draw=True):

        lmList = []
        if self.results.pose_landmarks:
            myPose = self.results.pose_landmarks

            for id, lm in enumerate(myPose.landmark):
                h, w, c = img.shape  # Get the height, width and channel of the image
                cx, cy = int(lm.x * w), int(lm.y * h)  # Convert the landmarks from normalized values to pixel values
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                # how to get index of each landmark
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 100), cv2.FILLED)

        return lmList



#Dummy main function can be used in any other file
def main():
    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture('pose vids/pose_2.mp4')
    detector = poseDectector()
    while True:
        success, img = cap.read()
        img  = detector.findPose(img)
        lmlist = detector.findPosition(img,  draw=False)
        if len(lmlist) != 0:
            print(lmlist[0])
        cv2.circle(img, (lmlist[0][1], lmlist[0][2]), 10, (255, 0, 0), cv2.FILLED)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255),
                3)  # display the fps on the image

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()