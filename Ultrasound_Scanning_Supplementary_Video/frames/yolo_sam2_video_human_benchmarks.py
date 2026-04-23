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
    avg_latency = np.mean(latencies)
    fps = 1000 / avg_latency
    stability_score = np.std(roi_areas) # Lower is more stable/less flickering
    
    return {
        "Model": model_path.split('/')[-1],
        "Avg Latency (ms)": round(avg_latency, 2),
        "FPS": round(fps, 1),
        "VRAM (MB)": round(vram_used, 2),
        "Stability (StdDev)": round(stability_score, 2)
    }

# Example of running it for all your models
# train2 for phantom data, train3 for human data (YOLOv8 nano)
# train6 for phantom data,train5 for human data (yolo11n)
# train7 for phantom data,train8 for human data (yolov10n)
# train9 for online_human data (yolov8n))
models_to_test = [
    "./runs/detect/train9/weights/best.pt", # v8n (with online human data)
    "./runs/detect/train3/weights/best.pt", # v8n (with human data)
    "./runs/detect/train8/weights/best.pt", # v10n (with human data)
    "./runs/detect/train5/weights/best.pt"  # v11n (with human data)
]

results_table = []
for m in models_to_test:
    results_table.append(run_benchmark(m, "/home/athena/Ultrasound_videos/human_test_results/Process_Data/Data_P_Human_Throat_US/ywc/filtered/20250930121841/20250930121841_shortlisted_v2.mp4"))

# Print comparison
print(f"{'Model':<20} | {'FPS':<6} | {'Latency':<8} | {'VRAM':<8} | {'Stability'}")
for r in results_table:
    print(f"{r['Model']:<20} | {r['FPS']:<6} | {r['Avg Latency (ms)']:<8} | {r['VRAM (MB)']:<8} | {r['Stability (StdDev)']}")