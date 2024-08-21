import numpy as np
from constants import FOREHEAD_INDICES, LEFT_CHEEK_INDICES, RIGHT_CHEEK_INDICES


def get_face_bbox(face_landmarks):
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


def get_forehead_contour(face_landmarks, image_shape):
    h, w, c = image_shape
    forehead_indices = FOREHEAD_INDICES
    forehead_landmarks = [face_landmarks[i] for i in forehead_indices]
    forehead_contour = [(int(landmark.x * w), int(landmark.y * h)) for landmark in
                        forehead_landmarks]
    return np.array(forehead_contour, dtype=np.int32)


def get_cheek_contours(face_landmarks, image_shape):
    h, w, c = image_shape
    left_cheek_landmarks = [face_landmarks[i] for i in LEFT_CHEEK_INDICES]
    right_cheek_landmarks = [face_landmarks[i] for i in RIGHT_CHEEK_INDICES]
    left_cheek_contour = [(int(landmark.x * w), int(landmark.y * h)) for landmark in left_cheek_landmarks]
    right_cheek_contour = [(int(landmark.x * w), int(landmark.y * h)) for landmark in right_cheek_landmarks]
    return np.array(left_cheek_contour, dtype=np.int32), np.array(right_cheek_contour, dtype=np.int32)
