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


def compute_optical_flow(prevFrame: np.ndarray, nextFrame: np.ndarray):
    prev_gray = cv.cvtColor(prevFrame, cv.COLOR_BGR2GRAY)
    prev_corners = cv.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
    next_gray = cv.cvtColor(nextFrame, cv.COLOR_BGR2GRAY)

    if prev_corners is None:
        return np.eye(3, dtype=np.float32)  # No corners detected, return identity transformation

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(
        winSize = (15, 15),
        maxLevel = 2,
        criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    next_corners, status, _ = cv.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_corners, None, **lk_params)

    valid = status.flatten() == 1
    prev_corners_good = prev_corners[valid]
    next_corners_good = next_corners[valid]

    if len(prev_corners_good) < 3:
        return np.eye(3, dtype=np.float32)  # Not enough points for transformation estimation

    transform_2x3, _ = cv.estimateAffinePartial2D(prev_corners_good, next_corners_good)
    if transform_2x3 is None:
        return np.eye(3, dtype=np.float32)

    transform_3x3 = np.vstack([transform_2x3, [0, 0, 1]]).astype(np.float32)
    return transform_3x3


