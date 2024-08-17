import cv2
from core import CoreModule
from utils import read_video, save_video, init_camera


def start():
    # 初始化摄像头
    cap = init_camera()
    core_module = CoreModule()
    timestamp = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # 处理图像
        image = core_module.process(frame, timestamp)
        timestamp += 1

        # 显示图像
        cv2.imshow('annotated_image', image)

        if cv2.waitKey(5) & 0xFF == 27:
            print("Closing Camera Stream")
            break

    cap.release()
    cv2.destroyAllWindows()


def start_video(video_path):
    if video_path is None:
        print("Please provide video path!")
        return
    video_frames = read_video(video_path)
    core_module = CoreModule(running_mode="VIDEO")
    output_video_frames = []
    for i, frame in enumerate(video_frames):
        print(f"Processing frame: {i}")
        image = core_module.process(frame, i)
        output_video_frames.append(image)

    save_video(output_video_frames)
    print("Video Processing Done!")


if __name__ == '__main__':
    video_path = "../asserts/video/sample2.mp4"
    start()
