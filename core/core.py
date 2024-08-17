import cv2
import mediapipe as mp
from detector import HandModule, FaceModule
from utils import draw_landmarks_on_hands, draw_landmarks_on_face, draw_points_trajectory, get_smooth_points, \
    get_face_bbox
from config import settings


class CoreModule:
    def __init__(self, running_mode="LIVE_STREAM"):
        self.running_mode = running_mode
        self.hand_module = None
        self.face_module = None
        self.hand_detector = None
        self.face_detector = None
        self.init_detector_module()

    def init_detector_module(self):
        self.hand_module = HandModule(self.running_mode)
        self.face_module = FaceModule(self.running_mode)
        self.hand_detector = self.hand_module.init_detector()
        self.face_detector = self.face_module.init_detector()

    def preprocess(self, frame):
        print("Preprocess...")
        image = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        image = cv2.flip(image, 1)
        return image

    def draw_landmarks(self, image, hand_result, face_result):
        image = draw_landmarks_on_hands(image, hand_result)
        image = draw_landmarks_on_face(image, face_result)
        return image

    def no_hand_or_face_hint(self, image, hand_result, face_result):
        if not hand_result.hand_landmarks:
            # 手移出时清空手部中心点数据
            self.hand_module.clear_hand_center_points()
            cv2.putText(image, 'No Hand Detected', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        elif not face_result.face_landmarks:
            cv2.putText(image, 'Face is Blocked', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    def is_hand_in_face(self, face_landmarks, hand_landmarks) -> bool:
        """
        检测手部是否在面部边框范围内，只要有任意一只在范围内就返回True
        :param face_landmarks:
        :param hand_landmarks:
        :return:
        """
        x_min, x_max, y_min, y_max = get_face_bbox(face_landmarks)
        for index, hand_landmark in enumerate(hand_landmarks):
            hand = hand_landmarks[index]
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
        if face_result.face_landmarks and hand_result.hand_landmarks:
            if self.is_hand_in_face(face_result.face_landmarks[0], hand_result.hand_landmarks):
                cv2.putText(image, 'Hand in Face Area', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, 'Hand not in Face Area', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    def process(self, frame, timestamp):
        # 图像预处理
        image = self.preprocess(frame)
        image_for_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        timestamp += 1

        # ------------- Detection Start ---------------#
        # 获取手部和面部关键点
        if self.running_mode == "LIVE_STREAM":
            print("Detecting for LIVE_STREAM...")
            self.hand_detector.detect_async(image_for_detect, timestamp)
            self.face_detector.detect_async(image_for_detect, timestamp)
        elif self.running_mode == "VIDEO":
            print("Detecting for VIDEO...")
            self.hand_module.result = self.hand_detector.detect_for_video(image_for_detect, timestamp)
            self.face_module.result = self.face_detector.detect_for_video(image_for_detect, timestamp)
        # ------------- Detection End -----------------#

        if self.hand_module.result and self.face_module.result:
            print("Get Hand and Face Result")
            hand_result = self.hand_module.result
            face_result = self.face_module.result

            # ------------- Operation Start ---------------#
            # 绘制手部和面部关键点
            if settings.DRAW_LANDMARKS:
                image = self.draw_landmarks(image, hand_result, face_result)

            # 提示无手部或者面部遮挡
            self.no_hand_or_face_hint(image, hand_result, face_result)

            # 判断手部位置是否在面部区域内
            self.hand_in_face_detection(image, hand_result, face_result)

            # 设置手部中心点数据
            self.hand_module.hand_center_detection(image, hand_result)

            # 显示手部中心点和轨迹
            image = self.hand_module.show_hand_center_point(image)
            # ------------- Operation End ---------------#
            return image
        else:
            return frame
