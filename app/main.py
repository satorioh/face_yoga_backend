import cv2
import mediapipe as mp
from collections import deque
from detector import HandModule, FaceModule
from utils import draw_landmarks_on_hands, draw_landmarks_on_face, get_hand_center_point, get_smooth_points
from config import settings


class CoreModule:
    def __init__(self):
        self.hand_module = None
        self.face_module = None
        self.hand_detector = None
        self.face_detector = None
        self.init_detector_module()
        self.hand_center_points_left = deque(maxlen=settings.FRAME_NUM_FOR_HAND_CENTER_POINTS)  # left
        self.hand_center_points_right = deque(maxlen=settings.FRAME_NUM_FOR_HAND_CENTER_POINTS)  # right

    def init_camera(self):
        cap = cv2.VideoCapture(settings.CAMERA_DEVICE)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        return cap

    def init_detector_module(self):
        self.hand_module = HandModule()
        self.face_module = FaceModule()
        self.hand_detector = self.hand_module.init_detector()
        self.face_detector = self.face_module.init_detector()

    def preprocess(self, frame):
        image = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        image = cv2.flip(image, 1)
        return image

    def draw_landmarks(self, image, hand_result, face_result):
        image = draw_landmarks_on_hands(image, hand_result)
        image = draw_landmarks_on_face(image, face_result)
        return image

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

    def loop_hand_face(self, image, face_landmarks, hand_landmarks, handedness) -> bool:
        """
        1.检测手部是否在面部边框范围内，只要有任意一只在范围内就返回True
        2.设置手部中心点数据
        :param face_landmarks:
        :param hand_landmarks:
        :return:
        """
        x_min, x_max, y_min, y_max = self.face_bbox(face_landmarks)
        for index, hand_landmark in enumerate(hand_landmarks):
            hand = hand_landmarks[index]
            hand_idx = handedness[index][0].index  # 0 for left, 1 for right
            hand_center_points_list = self.hand_center_points_left if hand_idx == 0 else self.hand_center_points_right

            # 设置手部中心点数据
            self.set_hand_center_point(image, hand, hand_center_points_list)

            # 检测手部是否在面部边框范围内
            for landmark in hand:
                if x_min < landmark.x < x_max and y_min < landmark.y < y_max:
                    return True
        return False

    def hand_in_face_detection(self, image, hand_result, face_result):
        """
        判断手部位置是否在面部区域内
        :param image:
        :param hand_result:
        :param face_result:
        :return:
        """
        if not hand_result.hand_landmarks:
            self.clear_hand_center_points()
            cv2.putText(image, 'No Hand Detected', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        elif face_result.face_landmarks and hand_result.hand_landmarks:
            if self.loop_hand_face(image, face_result.face_landmarks[0], hand_result.hand_landmarks,
                                   hand_result.handedness):
                cv2.putText(image, 'Hand in Face Area', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, 'Hand not in Face Area', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    def set_hand_center_point(self, image, hand, hand_center_points_list):
        h, w, c = image.shape
        center_x, center_y = get_hand_center_point(hand)
        hand_center_points_list.append((int(center_x * w), int(center_y * h)))

    def clear_hand_center_points(self):
        self.hand_center_points_left.clear()
        self.hand_center_points_right.clear()

    def show_hand_center_point(self, image):
        if len(self.hand_center_points_left) == settings.FRAME_NUM_FOR_HAND_CENTER_POINTS:
            smoothed_center_left = get_smooth_points(self.hand_center_points_left,
                                                     settings.FRAME_NUM_FOR_HAND_CENTER_POINTS)
            if smoothed_center_left:
                cv2.circle(image, (
                    int(smoothed_center_left[0]), int(smoothed_center_left[1])), 5,
                           (0, 0, 255), -1)

        if len(self.hand_center_points_right) == settings.FRAME_NUM_FOR_HAND_CENTER_POINTS:
            smoothed_center_right = get_smooth_points(self.hand_center_points_right,
                                                      settings.FRAME_NUM_FOR_HAND_CENTER_POINTS)
            if smoothed_center_right:
                cv2.circle(image, (
                    int(smoothed_center_right[0]), int(smoothed_center_right[1])), 5,
                           (255, 0, 0), -1)
        return image

    def start(self):
        cap = self.init_camera()
        timestamp = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # 图像预处理
            image = self.preprocess(frame)
            image_for_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            timestamp += 1

            # 获取手部和面部关键点
            self.hand_detector.detect_async(image_for_detect, timestamp)
            self.face_detector.detect_async(image_for_detect, timestamp)

            if self.hand_module.result and self.face_module.result:
                hand_result = self.hand_module.result
                face_result = self.face_module.result

                # 绘制手部和面部关键点
                if settings.DRAW_LANDMARKS:
                    image = self.draw_landmarks(image, hand_result, face_result)

                # 判断手部位置是否在面部区域内
                self.hand_in_face_detection(image, hand_result, face_result)

                # 显示手部中心点
                image = self.show_hand_center_point(image)

                # 显示图像
                cv2.imshow('annotated_image', image)

            if cv2.waitKey(5) & 0xFF == 27:
                print("Closing Camera Stream")
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    core_module = CoreModule()
    core_module.start()
