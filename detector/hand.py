import mediapipe as mp
from config import settings

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "../model/hand_landmarker.task"
RUNNING_MODE = settings.RUNNING_MODE


class HandModule:
    def __init__(self):
        self.result = None

    def init_detector(self):
        print("HandModule init_detector")
        base_options = BaseOptions(model_asset_path=MODEL_PATH, delegate=BaseOptions.Delegate.CPU)
        options = HandLandmarkerOptions(base_options=base_options, running_mode=VisionRunningMode.LIVE_STREAM,
                                        num_hands=2,
                                        result_callback=self.print_result)
        if RUNNING_MODE == "VIDEO":
            options = HandLandmarkerOptions(base_options=base_options, running_mode=VisionRunningMode.VIDEO,
                                            num_hands=2)
        return HandLandmarker.create_from_options(options)

    def print_result(self, result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.result = result
