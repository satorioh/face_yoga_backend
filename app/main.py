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

    def face_bbox(self, face_landmarks):
        """
        获取面部边框
        :param face_landmarks:
        :return:
        """
        x = [landmark.x for landmark in face_landmarks]
        y = [landmark.y for landmark in face_landmarks]
        x_min, x_max = min(x), max(x)
        y_min, y_max = min(y), max(y)
        return x_min, x_max, y_min, y_max

    def hand_in_face_bbox(self, face_landmarks, hand_landmarks):
        """
        检测手部是否在面部边框范围内，只要有任意一只在范围内就返回True
        :param face_landmarks:
        :param hand_landmarks:
        :return:
        """
        x_min, x_max, y_min, y_max = self.face_bbox(face_landmarks)
        hand_landmarks_len = len(hand_landmarks)
        for idx in range(hand_landmarks_len):
            hand = hand_landmarks[idx]
            for landmark in hand:
                if x_min < landmark.x < x_max and y_min < landmark.y < y_max:
                    return True
        return False

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

            if self.hand_module.result and self.face_module.result:
                hand_result = self.hand_module.result
                face_result = self.face_module.result
                annotated_hands_image = draw_landmarks_on_hands(image, hand_result)
                annotated_image = draw_landmarks_on_face(annotated_hands_image, face_result)
                if face_result.face_landmarks and hand_result.hand_landmarks:
                    if self.hand_in_face_bbox(face_result.face_landmarks[0], hand_result.hand_landmarks):
                        cv2.putText(annotated_image, 'Hand in Face Area', (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(annotated_image, 'Hand not in Face Area', (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('annotated_image', annotated_image)

            if cv2.waitKey(5) & 0xFF == 27:
                print("Closing Camera Stream")
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    core_module = CoreModule()
    core_module.start()
