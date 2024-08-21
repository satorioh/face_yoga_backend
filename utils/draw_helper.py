import cv2
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from constants import FOREHEAD_ARROW_INDEX, CHEEKS_ARROW_INDEX
from utils import point2pixel


def draw_landmarks_on_hands(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    hand_landmarks_list_len = len(hand_landmarks_list)
    for idx in range(hand_landmarks_list_len):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

    return annotated_image


def draw_landmarks_on_face(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())

    return annotated_image


def draw_points_trajectory(image, points):
    points_len = len(points)
    for i in range(1, points_len):
        cv2.line(image, points[i - 1], points[i], (0, 255, 0), 5)
    return image


def draw_arrows_on_forehead(image, face_landmarks):
    # Define arrow parameters
    arrow_color = (0, 255, 0)  # Green color
    arrow_thickness = 2
    arrow_tip_length = 0.3  # Relative size of the arrow tip

    # Calculate forehead region
    arrow_landmarks = [face_landmarks[i] for i in FOREHEAD_ARROW_INDEX]
    arrow_left_start, arrow_left_end, arrow_right_start, arrow_right_end = [point2pixel(landmark, image.shape) for
                                                                            landmark in arrow_landmarks]

    # Draw arrows
    cv2.arrowedLine(image, arrow_left_start, arrow_left_end, arrow_color, arrow_thickness, tipLength=arrow_tip_length)
    cv2.arrowedLine(image, arrow_right_start, arrow_right_end, arrow_color, arrow_thickness, tipLength=arrow_tip_length)

    return image


def draw_arrows_on_cheeks(image, face_landmarks):
    # Define arrow parameters
    arrow_color = (0, 255, 0)  # Green color
    arrow_thickness = 2
    arrow_tip_length = 0.3  # Relative size of the arrow tip

    # Calculate forehead region
    arrow_landmarks = [face_landmarks[i] for i in CHEEKS_ARROW_INDEX]
    arrow_left_start, arrow_left_end, arrow_right_start, arrow_right_end = [point2pixel(landmark, image.shape) for
                                                                            landmark in arrow_landmarks]

    # Draw arrows
    cv2.arrowedLine(image, arrow_left_start, arrow_left_end, arrow_color, arrow_thickness, tipLength=arrow_tip_length)
    cv2.arrowedLine(image, arrow_right_start, arrow_right_end, arrow_color, arrow_thickness, tipLength=arrow_tip_length)

    return image
