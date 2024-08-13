import mediapipe as mp
import cv2

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "../model/face_landmarker.task"


class FaceModule:
    def __init__(self):
        self.result = None

    def init_detector(self):
        print("FaceModule init_detector")
        base_options = BaseOptions(model_asset_path=MODEL_PATH, delegate=BaseOptions.Delegate.CPU)
        options = FaceLandmarkerOptions(base_options=base_options, running_mode=VisionRunningMode.LIVE_STREAM,
                                        result_callback=self.print_result)
        return FaceLandmarker.create_from_options(options)

    def print_result(self, result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.result = result
