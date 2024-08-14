import numpy as np


def get_hand_center_point(hand_landmarks):
    center_x = hand_landmarks[9].x
    center_y = hand_landmarks[9].y
    return center_x, center_y


def get_smooth_points(points, frame_num: int):
    if len(points) < frame_num:
        return None
    smoothed_x = np.mean([point[0] for point in points])
    smoothed_y = np.mean([point[1] for point in points])
    return smoothed_x, smoothed_y
