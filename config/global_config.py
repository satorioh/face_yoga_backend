import mediapipe as mp

VisionRunningMode = mp.tasks.vision.RunningMode


class Settings:
    # -------------camera 设置---------------#
    CAMERA_DEVICE = 0
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    # -------------显示设置----------------- #
    DRAW_LANDMARKS = True
    DRAW_HAND_CENTER_POINTS = False
    # -------------Mediapipe 设置---------------#
    RUNNING_MODE = "VIDEO"
    # -------------手部计算设置---------------#
    FRAME_NUM_FOR_HAND_CENTER_POINTS = 10


settings = Settings()
