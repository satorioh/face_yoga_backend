import cv2
import mediapipe as mp
from detector import HandModule, FaceModule
from utils import draw_landmarks_on_hands, draw_landmarks_on_face

CAMERA_DEVICE = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720


class CoreModule:
    def __init__(self):
        self.hand_module = None
        self.face_module = None
        self.hand_detector = None
        self.face_detector = None
        self.init_detector_module()

    def init_camera(self):
        cap = cv2.VideoCapture(CAMERA_DEVICE)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        return cap

    def init_detector_module(self):
        self.hand_module = HandModule()
        self.face_module = FaceModule()
        self.hand_detector = self.hand_module.init_detector()
        self.face_detector = self.face_module.init_detector()

    def start(self):
        cap = self.init_camera()
        timestamp = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            image = cv2.flip(image, 1)
            image_for_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            timestamp += 1
            self.hand_detector.detect_async(image_for_detect, timestamp)
            self.face_detector.detect_async(image_for_detect, timestamp)

            if self.face_module.result and self.hand_module.result:
                annotated_hands_image = draw_landmarks_on_hands(image, self.hand_module.result)
                annotated_image = draw_landmarks_on_face(annotated_hands_image, self.face_module.result)
                cv2.imshow('annotated_image', annotated_image)

            if cv2.waitKey(5) & 0xFF == 27:
                print("Closing Camera Stream")
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    core_module = CoreModule()
    core_module.start()
