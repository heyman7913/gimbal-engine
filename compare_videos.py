"""Compare stability metrics between two stabilized videos."""
import cv2
import numpy as np
import sys


def compute_stability_metrics(video_path, sample_every=1):
    """Compute stability metrics by measuring inter-frame motion."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"  Resolution: {width}x{height}, FPS: {fps:.1f}, Frames: {total_frames}")
    
    prev_gray = None
    motion_magnitudes = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % sample_every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_gray is not None:
                # Compute dense optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 
                    pyr_scale=0.5, levels=3, winsize=15, 
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                motion_magnitudes.append(np.mean(magnitude))
            
            prev_gray = gray
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...")
    
    cap.release()
    
    motion = np.array(motion_magnitudes)
    return {
        'frames': total_frames,
        'width': width,
        'height': height,
        'fps': fps,
        'mean_motion': np.mean(motion),
        'std_motion': np.std(motion),
        'max_motion': np.max(motion),
        'p95_motion': np.percentile(motion, 95),
        'stability_score': 1.0 / (1.0 + np.std(motion))
    }


def main():
    v3_path = "output_strongv3.mp4"
    v4_path = "output_v4_pure_cupy.mp4"
    
    # Sample every 2 frames to speed up analysis
    sample_every = 2
    
    print(f"\nAnalyzing {v3_path} (v3 - hybrid OpenCV+CuPy)...")
    v3 = compute_stability_metrics(v3_path, sample_every)
    if v3 is None:
        return
    
    print(f"\nAnalyzing {v4_path} (v4 - pure CuPy vectorized)...")
    v4 = compute_stability_metrics(v4_path, sample_every)
    if v4 is None:
        return
    
    print()
    print("=" * 65)
    print("STABILITY COMPARISON: v3 (hybrid) vs v4 (pure CuPy)")
    print("=" * 65)
    print(f"{'Metric':<25} {'v3':>12} {'v4':>12} {'Winner':>10}")
    print("-" * 65)
    
    def compare(name, v3_val, v4_val, lower_is_better=True):
        if lower_is_better:
            winner = "v3" if v3_val < v4_val else "v4" if v4_val < v3_val else "tie"
            diff = (v4_val - v3_val) / v3_val * 100 if v3_val != 0 else 0
        else:
            winner = "v3" if v3_val > v4_val else "v4" if v4_val > v3_val else "tie"
            diff = (v3_val - v4_val) / v4_val * 100 if v4_val != 0 else 0
        marker = "*" if winner != "tie" else ""
        print(f"{name:<25} {v3_val:>12.3f} {v4_val:>12.3f} {winner:>8}{marker}")
    
    compare("Mean Motion (px)", v3['mean_motion'], v4['mean_motion'], True)
    compare("Motion Std Dev", v3['std_motion'], v4['std_motion'], True)
    compare("95th Percentile (px)", v3['p95_motion'], v4['p95_motion'], True)
    compare("Max Motion (px)", v3['max_motion'], v4['max_motion'], True)
    compare("Stability Score", v3['stability_score'], v4['stability_score'], False)
    
    print("-" * 65)
    print(f"Frames analyzed: v3={v3['frames']}, v4={v4['frames']}")
    print()
    print("Lower motion values = more stable video")
    print("Higher stability score = better (computed as 1/(1+std))")


if __name__ == "__main__":
    main()
