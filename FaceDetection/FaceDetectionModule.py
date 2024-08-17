
import cv2
import mediapipe as mp
import time



class FaceDetector():
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #
        self.results = self.faceDetection.process(imgRGB)

        bboxs = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # mpDraw.draw_detection(img, detection) # Draw the face landmark and detections

                # Drawing the binding box by ourself and displaying it on the screen
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = (int(bboxC.xmin * iw), int(bboxC.ymin * ih),
                        int(bboxC.width * iw), int(bboxC.height * ih))
                bboxs.append([id, bbox, detection.score])
                img = self.fancyDraw(img, bbox)

                cv2.putText(img, f'detection: {int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        return img , bboxs

    def fancyDraw(self, img, bbox, l = 30, t = 10):
        x, y, w, h = bbox
        x1, y1 = x+w, y+h
        cv2.rectangle(img, bbox, (255, 0, 100), 2)
        cv2.line(img, (x, y), (x+l, y), (255, 0, 100), t)
        cv2.line(img, (x, y), (x, y+l), (255, 0, 100), t)
        cv2.line(img, (x1, y1), (x1-l, y1), (255, 0, 100), t)
        cv2.line(img, (x1,y1), (x1, y1-l), (255, 0, 100), t)

        return img
def main():
    pTime , cTime = 0, 0
    cap = cv2.VideoCapture('facevideos/face4.mp4')
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    detector = FaceDetector()

    while True:
        success, img = cap.read()
        img, bbox = detector.findFaces(img)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,f'FPS: {int(fps)}', (20,50), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(10)# 1 millisecond delay between frames

if __name__ == "__main__":
    main()