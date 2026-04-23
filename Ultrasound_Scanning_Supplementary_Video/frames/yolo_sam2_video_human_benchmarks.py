import time
import torch
import numpy as np
import cv2
import os
import csv
from glob import glob
from ultralytics import YOLO

def run_benchmark(model_path, video_path):
    # Fix: Define these at the start so they are always available for the return dict
    video_name = os.path.basename(video_path)
    model_name = os.path.basename(model_path)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load model
    model = YOLO(model_path).to(device)
    
    # Track VRAM
    vram_used = torch.cuda.memory_allocated(device) / (1024**2) 
    
    cap = cv2.VideoCapture(video_path)
    latencies = []
    roi_areas = []
    
    print(f"Benchmarking: {model_name} on {video_name}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        start_time = time.perf_counter()
        results = model.predict(frame, conf=0.5, verbose=False)[0]
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        
        if len(results.boxes) > 0:
            # Track the first box's area for stability
            box = results.boxes.xywh[0]
            area = box[2] * box[3]
            roi_areas.append(area.item())

    cap.release()
    
    # Calculate Metrics
    avg_latency = np.mean(latencies) if latencies else 0
    fps = 1000 / avg_latency if avg_latency > 0 else 0
    stability_score = np.std(roi_areas) if len(roi_areas) > 1 else float('nan')
    
    return {
        "Video": video_name,
        "Model": model_path.split('/')[-3],  # Shows 'train9', 'train10', etc.
        "Avg Latency (ms)": round(avg_latency, 2),
        "FPS": round(fps, 1),
        "VRAM (MB)": round(vram_used, 2),
        "Stability (StdDev)": round(stability_score, 2)
    }

video_folder = "/home/athena/Ultrasound_videos/online_videos/"
video_files = sorted(glob(os.path.join(video_folder, "*.mp4")))

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
csv_output = "benchmark_results.csv"

print(f"Found {len(video_files)} videos. Starting benchmarks...")

for video_path in video_files:
    v_name = os.path.basename(video_path)
    print(f"\n>>> Video: {v_name}")
    print(f"{'Model':<10} | {'FPS':<6} | {'Lat (ms)':<8} | {'Stability'}")
    
    for model_path in models_to_test:
        try:
            res = run_benchmark(model_path, video_path)
            all_results.append(res)
            print(f"{res['Model']:<10} | {res['FPS']:<6} | {res['Avg Latency (ms)']:<8} | {res['Stability (StdDev)']}")
        except Exception as e:
            print(f"Error testing {model_path} on {v_name}: {e}")

# --- SAVE TO CSV ---
if all_results:
    keys = all_results[0].keys()
    with open(csv_output, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(all_results)
    print(f"\nDetailed results saved to: {csv_output}")

# --- FINAL SUMMARY ---
print("\n" + "="*20 + " OVERALL MODEL AVERAGES " + "="*20)
summary = {}
for r in all_results:
    m = r['Model']
    if m not in summary: summary[m] = {"s": [], "f": []}
    if not np.isnan(r['Stability (StdDev)']):
        summary[m]["s"].append(r['Stability (StdDev)'])
    summary[m]["f"].append(r['FPS'])

print(f"{'Model':<10} | {'Avg FPS':<8} | {'Avg Stability'}")
for m, data in summary.items():
    m_fps = np.mean(data["f"]) if data["f"] else 0
    m_stab = np.mean(data["s"]) if data["s"] else float('nan')
    print(f"{m:<10} | {m_fps:<8.1f} | {m_stab:.2f}")