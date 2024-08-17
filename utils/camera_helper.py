import cv2
from config import settings


def init_camera():
    print("Init Camera...")
    cap = cv2.VideoCapture(settings.CAMERA_DEVICE)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.CAMERA_WIDTH)
    return cap
