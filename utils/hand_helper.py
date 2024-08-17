import numpy as np


def get_hand_center_point(hand_landmarks):
    center_x = hand_landmarks[9].x
    center_y = hand_landmarks[9].y
    return center_x, center_y


def get_hand_contour(hand_landmarks, image_shape):
    h, w, c = image_shape
    hand_contour = [(int(landmark.x * w), int(landmark.y * h)) for landmark in hand_landmarks]
    return np.array(hand_contour, dtype=np.int32)
