import cv2
import mediapipe as mp
import time

class FaceMeshDetector:

    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode= self.staticMode,max_num_faces= self.maxFaces,
                                                 min_detection_confidence = self.minDetectionCon,min_tracking_confidence= self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)

    def findFeshMesh(self,img, draw= True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        # print(self.results.multi_face_landmarks)
        facelm = []
        face =[]
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)

                face =[]
                for id, lm in enumerate(faceLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.putText(img, f'{str(id)}', (cx, cy), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 0), 1)
                    face.append([ cx, cy])
                facelm.append(face)
        return img , facelm

    # def fancyDraw(self, img, draw = True):

def main():
    cap = cv2.VideoCapture('facevideos/face4.mp4')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    pTime = 0
    detector = FaceMeshDetector()

    while True:
        success, img = cap.read()
        img, faces = detector.findFeshMesh(img)
        if(len(faces[0]) != 0):
            print(len(faces[0]))
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        cv2.imshow('Image', img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()