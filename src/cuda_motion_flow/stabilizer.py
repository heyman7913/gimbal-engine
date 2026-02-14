from pathlib import Path
import cv2 as cv
import numpy as np
import cupy as cp

def stabilize_video(input_path: Path, output_path: Path, smoothing_factor: float, verbose: bool):
    video_capture = cv.VideoCapture(str(input_path))

    if not video_capture.isOpened():
        raise IOError(f"Cannot open video file: {input_path}")

    frame_width = int(video_capture.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv.CAP_PROP_FRAME_COUNT))

    if verbose:
        print(f"Video Properties:\n- Resolution: {frame_width}x{frame_height}\n- FPS: {fps}\n- Total Frames: {total_frames}")

    frames = []
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frames.append(frame)
    frames = np.array(frames)

    video_writer = cv.VideoWriter(str(output_path), cv.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height), True)

    for i, frame in enumerate(frames):
        video_writer.write(frame)
        if verbose and i % 30 == 0:
            print(f"Processed {i}/{total_frames} frames...")


    video_writer.release()
    video_capture.release()






