import cv2
from config import settings


def read_video(video_path):
    print("Reading video...")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_video(video_frames, output_video_path=settings.DEFAULT_OUTPUT_VIDEO_PATH):
    print("Saving video...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w = video_frames[0].shape[:2]
    out = cv2.VideoWriter(output_video_path, fourcc, settings.VIDEO_FPS, (w, h))
    for frame in video_frames:
        out.write(frame)
    out.release()
    print(f"Video saved at {output_video_path}")
