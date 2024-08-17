import cv2
import mediapipe as mp
from collections import deque
from config import settings
from utils import get_hand_center_point, get_smooth_points, draw_points_trajectory

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "../model/hand_landmarker.task"


class HandModule:
    def __init__(self, running_mode="LIVE_STREAM"):
        self.result = None
        self.running_mode = running_mode
        self.hand_center_points_left = deque(maxlen=settings.FRAME_NUM_FOR_HAND_CENTER_POINTS)  # left
        self.hand_center_points_right = deque(maxlen=settings.FRAME_NUM_FOR_HAND_CENTER_POINTS)  # right

    def init_detector(self):
        print("HandModule init_detector")
        base_options = BaseOptions(model_asset_path=MODEL_PATH, delegate=BaseOptions.Delegate.CPU)
        options = HandLandmarkerOptions(base_options=base_options, running_mode=VisionRunningMode.LIVE_STREAM,
                                        num_hands=2,
                                        result_callback=self.print_result)
        if self.running_mode == "VIDEO":
            options = HandLandmarkerOptions(base_options=base_options, running_mode=VisionRunningMode.VIDEO,
                                            num_hands=2)
        return HandLandmarker.create_from_options(options)

    def print_result(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.result = result

    def set_hand_center_point(self, image, hand, hand_center_points_list):
        h, w, c = image.shape
        center_x, center_y = get_hand_center_point(hand)
        hand_center_points_list.append((int(center_x * w), int(center_y * h)))

    def clear_hand_center_points(self):
        self.hand_center_points_left.clear()
        self.hand_center_points_right.clear()

    def hand_center_detection(self, image, hand_result):
        """
        判断左右手并设置手部中心点数据
        :param image:
        :param hand_result:
        :return:
        """
        hand_landmarks = hand_result.hand_landmarks
        handedness = hand_result.handedness
        hand_landmarks_len = len(hand_landmarks)

        if hand_landmarks_len == 1:
            # 一只手时看handedness : 0 for right, 1 for left
            hand_idx = handedness[0][0].index
            hand_center_points_list = self.hand_center_points_right if hand_idx == 0 else self.hand_center_points_left
            another_hand_center_points_list = self.hand_center_points_left if hand_idx == 0 else self.hand_center_points_right
            # 只有一只手，说明另一只手移出了
            another_hand_center_points_list.clear()
            # 设置手部中心点数据
            self.set_hand_center_point(image, hand_landmarks[0], hand_center_points_list)

        elif hand_landmarks_len == 2:
            for index, hand_landmark in enumerate(hand_landmarks):
                hand = hand_landmarks[index]
                # 两只手时看index : 0 for left, 1 for right
                hand_center_points_list = self.hand_center_points_left if index == 0 else self.hand_center_points_right
                # 设置手部中心点数据
                self.set_hand_center_point(image, hand, hand_center_points_list)

    def show_hand_center_point(self, image):
        if len(self.hand_center_points_left) == settings.FRAME_NUM_FOR_HAND_CENTER_POINTS:
            smoothed_center_left = get_smooth_points(self.hand_center_points_left,
                                                     settings.FRAME_NUM_FOR_HAND_CENTER_POINTS)
            if smoothed_center_left:
                # 绘制手部中心点
                if settings.DRAW_HAND_CENTER_POINTS:
                    cv2.circle(image, (
                        int(smoothed_center_left[0]), int(smoothed_center_left[1])), 5,
                               (0, 255, 0), -1)

                # 绘制手部中心点轨迹
                draw_points_trajectory(image, self.hand_center_points_left)

        if len(self.hand_center_points_right) == settings.FRAME_NUM_FOR_HAND_CENTER_POINTS:
            smoothed_center_right = get_smooth_points(self.hand_center_points_right,
                                                      settings.FRAME_NUM_FOR_HAND_CENTER_POINTS)
            if smoothed_center_right:
                # 绘制手部中心点
                if settings.DRAW_HAND_CENTER_POINTS:
                    cv2.circle(image, (
                        int(smoothed_center_right[0]), int(smoothed_center_right[1])), 5,
                               (0, 255, 0), -1)

                # 绘制手部中心点轨迹
                draw_points_trajectory(image, self.hand_center_points_right)
        return image
