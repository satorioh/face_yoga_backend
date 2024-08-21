import mediapipe as mp

VisionRunningMode = mp.tasks.vision.RunningMode


class Settings:
    # -------------camera 设置---------------#
    CAMERA_DEVICE = 0
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    # -------------显示设置----------------- #
    DRAW_LANDMARKS = False
    DRAW_HAND_CENTER_POINTS = False
    DRAW_HAND_CONTOUR = False
    DRAW_FOREHEAD_CONTOUR = False
    DRAW_CHEEKS_CONTOUR = False
    # -------------手部计算设置---------------#
    FRAME_NUM_FOR_HAND_CENTER_POINTS = 10
    # -------------视频输出设置---------------#
    DEFAULT_OUTPUT_VIDEO_PATH = "../dist/output.mp4"
    VIDEO_FPS = 24


settings = Settings()
