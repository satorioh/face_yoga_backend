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
