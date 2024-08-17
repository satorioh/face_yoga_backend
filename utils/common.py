import cv2
import numpy as np


def point2pixel(point, image_shape):
    h, w, _ = image_shape
    return int(point.x * w), int(point.y * h)


def get_smooth_points(points, frame_num: int):
    if len(points) < frame_num:
        return None
    smoothed_x = np.mean([point[0] for point in points], dtype=np.float32)
    smoothed_y = np.mean([point[1] for point in points], dtype=np.float32)
    return smoothed_x, smoothed_y


def get_contour_area(contour):
    return cv2.contourArea(contour)


def find_contour_hull(contour):
    return cv2.convexHull(contour)


def calculate_intersection_area(contour1, contour2):
    intersection, _ = cv2.intersectConvexConvex(contour1, contour2)
    return intersection
