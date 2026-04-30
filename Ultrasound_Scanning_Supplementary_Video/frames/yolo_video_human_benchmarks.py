import time
import torch
import numpy as np
import cv2
import os
import csv
from glob import glob
from ultralytics import YOLO
from medpy.metric.binary import dc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_benchmark(model_path, video_path, gt_folder):
    video_name = os.path.basename(video_path)
    v_base_name = os.path.splitext(video_name)[0]
    model_id = model_path.split('/')[-3]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path).to(device)
    
    cap = cv2.VideoCapture(video_path)
    latencies, roi_areas, dice_scores = [], [], []
    
    # Get all GT filenames for this video once to speed up matching
    all_gt_files = glob(os.path.join(gt_folder, f"{v_base_name}_*.png"))
    gt_frame_numbers = [int(f.split('_')[-1].split('.')[0]) for f in all_gt_files]
    
    current_frame = 0 
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 1. Inference
        start_time = time.perf_counter()
        results = model.predict(frame, conf=0.5, verbose=False)[0]
        latencies.append((time.perf_counter() - start_time) * 1000)
        
        # 2. Mask Creation
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if len(results.boxes) > 0:
            box = results.boxes.xyxy[0].cpu().numpy().astype(int)
            mask[box[1]:box[3], box[0]:box[2]] = 1
            roi_areas.append((box[2]-box[0]) * (box[3]-box[1]))

        # 3. Robust Accuracy Comparison
        # Instead of matching current_frame, we look for any GT file that 
        # actually exists in your 90-frame set for this video.
        # We check the folder for the specific frame number suffixes you mentioned.
        match = [f for f in all_gt_files if f.endswith(f"_{current_frame:05d}.png") or 
                                           f.endswith(f"_{current_frame:03d}.png") or
                                           f.endswith(f"_{current_frame:01d}.png")]

        if match:
            gt_img = cv2.imread(match[0], 0)
            gt_mask = np.where(gt_img > 0, 1, 0).astype(np.uint8)
            dice_scores.append(dc(mask, gt_mask))
            print(f"    Matched GT: {os.path.basename(match[0])} at Video Frame {current_frame}")
        
        current_frame += 1

    cap.release()
    return {
        "Video": video_name, "Model": model_id,
        "Avg Latency (ms)": round(np.mean(latencies), 2),
        "FPS": round(1000 / np.mean(latencies), 1) if latencies else 0,
        "Mean Dice": round(np.mean(dice_scores), 4) if dice_scores else 0,
        "Stability (StdDev)": round(np.std(roi_areas), 2) if roi_areas else 0
    }

def visualize_results(csv_path):
    df = pd.read_csv(csv_path)
    plot_df = df[df['Mean Dice'] > 0] # Only plot videos where GT was found
    
    if plot_df.empty:
        print("Warning: No Dice scores > 0 found. Check your GT naming!")
        return

    sns.set_theme(style="whitegrid")
    
    # Figure 1: Mean Dice Comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x='Video', y='Mean Dice', hue='Model')
    plt.xticks(rotation=45, ha='right')
    plt.title('Accuracy Comparison: Mean Dice Score per Video')
    plt.tight_layout()
    plt.savefig('dice_comparison.png')
    
    # Figure 2: Efficiency Comparison (FPS vs Dice)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=plot_df, x='FPS', y='Mean Dice', hue='Model', s=100)
    plt.title('Model Efficiency: FPS vs. Accuracy')
    plt.tight_layout()
    plt.savefig('efficiency_tradeoff.png')
    print("Visualizations saved as 'dice_comparison.png' and 'efficiency_tradeoff.png'.")
# --- CONFIGURATION ---
models_to_test = [
    "./runs/detect/train9/weights/best.pt",  # v8n online human
    "./runs/detect/train10/weights/best.pt", # v10n online human
    "./runs/detect/train11/weights/best.pt", # v11n online human
    "./runs/detect/train3/weights/best.pt",  # v8n human
    "./runs/detect/train8/weights/best.pt",  # v10n human
    "./runs/detect/train5/weights/best.pt"   # v11n human
]

video_folder = "/home/athena/Ultrasound_videos/Ultrasound_Scanning_Supplementary_Video/frames/all_online_human_videos/"
gt_folder = "/home/athena/Ultrasound_videos/Ultrasound_Scanning_Supplementary_Video/frames/online_human_selected_GT_label/SegmentationClass/" 

# --- EXECUTION ---
all_results = []
video_files = sorted(glob(os.path.join(video_folder, "*.mp4")))

for video_path in video_files:
    print(f"Processing: {os.path.basename(video_path)}")
    for model_path in models_to_test:
        try:
            res = run_benchmark(model_path, video_path, gt_folder)
            all_results.append(res)
            print(f"  {res['Model']} | Dice: {res['Mean Dice']} | FPS: {res['FPS']}")
        except Exception as e: print(f"  Error: {e}")

if all_results:
    output_file = "yolo_benchmark_results.csv"
    pd.DataFrame(all_results).to_csv(output_file, index=False)
    visualize_results(output_file)