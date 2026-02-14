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

    transforms = []
    ret, prev_frame = video_capture.read()
    # Store the first frame for later use in stabilization
    if not ret:
        raise IOError("Failed to read the first frame of the video.")
    first_frame = prev_frame.copy()
    for i in range(1, total_frames):
        ret, next_frame = video_capture.read()
        if not ret:
            raise IOError(f"Failed to read frame {i} of the video.")
        transform = compute_optical_flow(prev_frame, next_frame)
        transforms.append(transform)
        prev_frame = next_frame
        if verbose and i % 30 == 0:
            print(f"Computed LK optical flow for {i}/{total_frames} frames...")
    transforms_arr = np.array(transforms, dtype=np.float32)
    print(f"Completed optical flow computation for all frames. Total transforms computed: {len(transforms)}")

    smoothed_transforms = smooth_trajectory(transforms_arr, smoothing_factor)

    video_writer = cv.VideoWriter(str(output_path), cv.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height), True)


    video_capture.release()
    video_capture = cv.VideoCapture(str(input_path))
    ret, first_frame = video_capture.read()
    if not ret:
        raise IOError("Failed to read the first frame during stabilization.")
    video_writer.write(first_frame)

    for i in range(1, total_frames):
        ret, frame = video_capture.read()
        if not ret:
            raise IOError(f"Failed to read frame {i} during stabilization.")
        M = smoothed_transforms[i-1][:2, :]
        stabilized_frame = cv.warpAffine(frame, M, (frame_width, frame_height), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
        video_writer.write(stabilized_frame)
        if verbose and i % 30 == 0:
            print(f"Applied stabilization to {i}/{total_frames} frames...")

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

def smooth_trajectory(transforms: np.ndarray, smoothing_factor: float):
    # Extracting frame-to-frame transformations
    dx = transforms[:, 0, 2]
    dy = transforms[:, 1, 2]
    da = np.arctan2(transforms[:, 1, 0], transforms[:, 0, 0])

    # Cumulative sum to get the trajectory
    traj_x = np.cumsum(dx)
    traj_y = np.cumsum(dy)
    traj_a = np.cumsum(da)

    # Window size for smoothing is determined via smoothing factor
    # The smoothing factor is clamped between 0.0 and 1.0 to ensure valid window sizes
    # Lowest - 5 frames (minimal smoothing), Highest - 101 frames (aggressive smoothing)
    # Created low-pass filter kernel for trajectory smoothing
    smoothing_factor = max(0.0, min(smoothing_factor, 1.0))
    window = int(5 + smoothing_factor * 96)
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window

    # Padded moving average convolution
    pad = window // 2
    traj_x_smoothed = np.convolve(np.pad(traj_x, (pad, pad), mode='edge'), kernel, mode='valid')
    traj_y_smoothed = np.convolve(np.pad(traj_y, (pad, pad), mode='edge'), kernel, mode='valid')
    traj_a_smoothed = np.convolve(np.pad(traj_a, (pad, pad), mode='edge'), kernel, mode='valid')

    corr_x = traj_x_smoothed - traj_x
    corr_y = traj_y_smoothed - traj_y
    corr_a = traj_a_smoothed - traj_a

    # Corrected transforms
    corrected = []
    for i in range(len(transforms)):
        a = da[i] + corr_a[i]
        cos_a = np.cos(a)
        sin_a = np.sin(a)
        T = np.array([
            [cos_a, -sin_a, dx[i] + corr_x[i]],
            [sin_a,  cos_a, dy[i] + corr_y[i]],
            [0,      0,     1]
        ], dtype=np.float32)
        corrected.append(T)

    return np.array(corrected, dtype=np.float32)

