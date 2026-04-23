import time
import torch
import numpy as np
import cv2
from ultralytics import YOLO

def run_benchmark(model_path, video_path):
    # 1. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path).to(device)
    
    # 2. Track VRAM (after model is loaded)
    vram_used = torch.cuda.memory_allocated(device) / (1024**2) # Convert to MB
    
    cap = cv2.VideoCapture(video_path)
    latencies = []
    roi_areas = []
    
    print(f"Benchmarking: {model_path}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # --- Start Timer ---
        start_time = time.perf_counter()
        
        # Run Inference
        results = model.predict(frame, conf=0.5, verbose=False)[0]
        
        # --- End Timer ---
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        
        # Track ROI for Stability (Example: Class 0)
        if len(results.boxes) > 0:
            # Using box area as a proxy for stability in this example
            box = results.boxes.xywh[0]
            area = box[2] * box[3]
            roi_areas.append(area.item())

    cap.release()
    
    # --- Calculate Final Metrics ---
    avg_latency = np.mean(latencies) if latencies else 0
    fps = 1000 / avg_latency if avg_latency > 0 else 0
    stability_score = np.std(roi_areas) if len(roi_areas) > 1 else float('nan')
    
    return {
        "Video": video_name,
        "Model": model_name,
        "Avg Latency (ms)": round(avg_latency, 2),
        "FPS": round(fps, 1),
        "VRAM (MB)": round(vram_used, 2),
        "Stability (StdDev)": round(stability_score, 2)
    }

video_folder = "/home/athena/Ultrasound_videos/online_videos/"
video_files = glob(os.path.join(video_folder, "*.mp4"))

# Example of running it for all your models
# train2 for phantom data, train3 for human data (YOLOv8 nano)
# train6 for phantom data,train5 for human data (yolo11n)
# train7 for phantom data,train8 for human data (yolov10n)
# train9 for online_human data (yolov8n))
# train10 from online_human data (yolov10n)
# train11 from online_human data (yolo11n)
models_to_test = [
    "./runs/detect/train9/weights/best.pt", # v8n (with online human data)
    "./runs/detect/train10/weights/best.pt", # v10n (with online human data)
    "./runs/detect/train11/weights/best.pt", # v11n (with online human data)
    "./runs/detect/train3/weights/best.pt", # v8n (with human data)
    "./runs/detect/train8/weights/best.pt", # v10n (with human data)
    "./runs/detect/train5/weights/best.pt"  # v11n (with human data)
]



# ------------------------- test single video -------------------------
# results_table = []
# for m in models_to_test:
#     results_table.append(run_benchmark(m, "/home/athena/Ultrasound_videos/human_test_results/Process_Data/Data_P_Human_Throat_US/ywc/filtered/20250930121841/20250930121841_shortlisted_v2.mp4"))
#     # results_table.append(run_benchmark(m, "/home/athena/Ultrasound_videos/online_videos/video2.mp4"))

# # Print comparison
# print(f"{'Model':<20} | {'FPS':<6} | {'Latency':<8} | {'VRAM':<8} | {'Stability'}")
# for r in results_table:
#     print(f"{r['Model']:<20} | {r['FPS']:<6} | {r['Avg Latency (ms)']:<8} | {r['VRAM (MB)']:<8} | {r['Stability (StdDev)']}")


# ------------------------- test all videos in a folder-------------------------
all_results = []

print(f"Found {len(video_files)} videos in {video_folder}")
print("-" * 80)

for video_path in video_files:
    print(f"\n>>> Processing Video: {os.path.basename(video_path)}")
    print(f"{'Model':<15} | {'FPS':<6} | {'Latency':<8} | {'Stability'}")
    
    for model_path in models_to_test:
        res = run_benchmark(model_path, video_path)
        all_results.append(res)
        
        # Print individual model result for the current video
        print(f"{res['Model']:<15} | {res['FPS']:<6} | {res['Avg Latency (ms)']:<8} | {res['Stability (StdDev)']}")

# --- FINAL SUMMARY ---
print("\n" + "="*30 + " FINAL SUMMARY " + "="*30)
# Grouping by model to see average performance across all videos
model_performance = {}
for r in all_results:
    m = r['Model']
    if m not in model_performance:
        model_performance[m] = {"stability": [], "fps": []}
    if not np.isnan(r['Stability (StdDev)']):
        model_performance[m]["stability"].append(r['Stability (StdDev)'])
    model_performance[m]["fps"].append(r['FPS'])

print(f"{'Model':<15} | {'Mean FPS':<10} | {'Mean Stability'}")
for m, metrics in model_performance.items():
    mean_fps = round(np.mean(metrics["fps"]), 1)
    mean_stab = round(np.mean(metrics["stability"]), 2) if metrics["stability"] else "N/A"
    print(f"{m:<15} | {mean_fps:<10} | {mean_stab}")