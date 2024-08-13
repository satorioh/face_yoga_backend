import mediapipe as mp
import cv2

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "../model/hand_landmarker.task"


class HandModule:
    def __init__(self):
        self.result = None

    def init_detector(self):
        print("HandModule init_detector")
        base_options = BaseOptions(model_asset_path=MODEL_PATH, delegate=BaseOptions.Delegate.CPU)
        options = HandLandmarkerOptions(base_options=base_options, running_mode=VisionRunningMode.LIVE_STREAM,
                                        num_hands=2,
                                        result_callback=self.print_result)
        return HandLandmarker.create_from_options(options)

    def print_result(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.result = result
