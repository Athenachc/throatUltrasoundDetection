import time
import torch
import numpy as np
import cv2
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys 
from glob import glob
from ultralytics import YOLO
from medpy.metric.binary import dc

# Hydra and SAM2 specific imports
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- CONFIGURATION & PATHS ---
yolo_models_list = [
    "./runs/detect/train9/weights/best.pt",  # v8n online human
    "./runs/detect/train10/weights/best.pt", # v10n online human
    "./runs/detect/train11/weights/best.pt", # v11n online human
    "./runs/detect/train3/weights/best.pt",  # v8n human
    "./runs/detect/train8/weights/best.pt",  # v10n human
    "./runs/detect/train5/weights/best.pt"   # v11n human
]

video_folder = "/home/athena/Ultrasound_videos/Ultrasound_Scanning_Supplementary_Video/frames/all_online_human_videos/"
gt_folder = "/home/athena/Ultrasound_videos/Ultrasound_Scanning_Supplementary_Video/frames/online_human_selected_GT_label/SegmentationClass/" 

# --- HELPER FUNCTIONS ---

def print_progress(current, total, video_name):
    percent = (current / total) * 100
    msg = f"\r    >>> [{video_name}] Frame: {current}/{total} ({percent:.1f}%) "
    sys.stderr.write(msg)
    sys.stderr.flush()

def save_visual_comparison(frame_rgb, gt_mask, model_masks, video_name, frame_idx):
    full_data_models = ['train9', 'train10', 'train11']
    limited_data_models = ['train3', 'train5', 'train8']
    
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 5)

    # Row 1: Original, GT, and Full Train Models
    ax_orig = fig.add_subplot(gs[0, 0]); ax_orig.imshow(frame_rgb); ax_orig.set_title(f"Original\n({video_name} @ {frame_idx})"); ax_orig.axis('off')
    ax_gt = fig.add_subplot(gs[0, 1]); ax_gt.imshow(gt_mask, cmap='gray'); ax_gt.set_title("Ground Truth"); ax_gt.axis('off')

    for i, m_id in enumerate(full_data_models):
        ax = fig.add_subplot(gs[0, i+2])
        ax.imshow(frame_rgb, alpha=0.6)
        if m_id in model_masks: ax.imshow(model_masks[m_id], cmap='jet', alpha=0.4)
        ax.set_title(f"{m_id} (Full)"); ax.axis('off')

    # Row 2: Limited Data Models
    for i, m_id in enumerate(limited_data_models):
        ax = fig.add_subplot(gs[1, i+1])
        ax.imshow(frame_rgb, alpha=0.6)
        if m_id in model_masks: ax.imshow(model_masks[m_id], cmap='jet', alpha=0.4)
        ax.set_title(f"{m_id} (No Online)"); ax.axis('off')

    plt.tight_layout()
    os.makedirs("validation_visuals", exist_ok=True)
    save_path = f"validation_visuals/{video_name}_frame_{frame_idx:05d}_comparison.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def plot_benchmark_results(results_list):
    df = pd.DataFrame(results_list)
    if df.empty: return
    summary = df.groupby('Model').agg({'Mean Dice': 'mean', 'FPS': 'mean', 'Avg Latency (ms)': 'mean'}).reset_index()
    sns.set_theme(style="whitegrid")

    # Dice Comparison
    plt.figure(figsize=(10, 6))
    colors = ["#2ecc71" if m in ['train9', 'train10', 'train11'] else "#e74c3c" for m in summary['Model']]
    ax = sns.barplot(data=summary, x='Model', y='Mean Dice', palette=colors)
    plt.title('Mean Dice Score Comparison'); plt.ylim(0, 1.0)
    plt.savefig("dice_comparison.png", dpi=300, bbox_inches='tight'); plt.close()

    # Efficiency Trade-off
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=summary, x='FPS', y='Mean Dice', hue='Model', s=200)
    plt.title('Efficiency vs. Accuracy Trade-off')
    plt.savefig("efficiency_tradeoff.png", dpi=300, bbox_inches='tight'); plt.close()

# --- CORE BENCHMARK ENGINE ---

def run_hybrid_benchmark(loaded_yolos, sam2_predictor, video_path, gt_folder):
    video_name = os.path.basename(video_path)
    v_base_name = os.path.splitext(video_name)[0]
    cap = cv2.VideoCapture(video_path)
    
    # 1. Identify GT frames using Regex (handles 01, 001, 00001 padding)
    all_gt_paths = glob(os.path.join(gt_folder, f"{v_base_name}_*.png"))
    target_frames = {}
    for path in all_gt_paths:
        match = re.search(r'_(\d+)\.png$', path)
        if match:
            target_frames[int(match.group(1))] = path

    stats = {m_id: {"latencies": [], "dice": []} for m_id in loaded_yolos.keys()}
    
    # 2. Process only relevant frames
    for frame_idx in sorted(target_frames.keys()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: # Try 1-based indexing fallback
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
            ret, frame = cap.read()
            if not ret: continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gt_mask = np.where(cv2.imread(target_frames[frame_idx], 0) > 0, 1, 0).astype(np.uint8)
        current_masks = {}

        for m_id, model in loaded_yolos.items():
            start = time.perf_counter()
            res = model.predict(frame, conf=0.5, verbose=False)[0]
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            
            if len(res.boxes) > 0:
                sam2_predictor.set_image(frame_rgb)
                for box in res.boxes.xyxy.cpu().numpy():
                    m, _, _ = sam2_predictor.predict(box=box, multimask_output=False)
                    mask = np.maximum(mask, m[0].astype(np.uint8))
            
            stats[m_id]["latencies"].append((time.perf_counter() - start) * 1000)
            stats[m_id]["dice"].append(dc(mask, gt_mask))
            current_masks[m_id] = mask

        save_visual_comparison(frame_rgb, gt_mask, current_masks, v_base_name, frame_idx)
        print(f"    [Processed] {v_base_name} Frame {frame_idx}")

    cap.release()
    return [{
        "Video": video_name, "Model": m, "Avg Latency (ms)": round(np.mean(d["latencies"]), 2),
        "FPS": round(1000/np.mean(d["latencies"]), 1), "Mean Dice": round(np.mean(d["dice"]), 4)
    } for m, d in stats.items() if d["dice"]]

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    print("Loading YOLO models...")
    loaded_yolos = {path.split('/')[-3]: YOLO(path).to("cuda") for path in yolo_models_list}

    print("Initializing SAM2...")
    GlobalHydra.instance().clear()
    initialize(config_path=".", version_base=None)
    sam2_predictor = SAM2ImagePredictor(build_sam2("sam2.1_hiera_l", "/home/athena/sam2/checkpoints/sam2.1_hiera_large.pt", device="cuda"))

    all_results = []
    video_files = sorted(glob(os.path.join(video_folder, "*.mp4")))

    for video_path in video_files:
        print(f"\n--- Video: {os.path.basename(video_path)} ---")
        try:
            all_results.extend(run_hybrid_benchmark(loaded_yolos, sam2_predictor, video_path, gt_folder))
        except Exception as e:
            print(f"Error: {e}")

    if all_results:
        pd.DataFrame(all_results).to_csv("hybrid_benchmark_results.csv", index=False)
        plot_benchmark_results(all_results)
        print("\nSuccess. Check 'hybrid_benchmark_results.csv' and 'validation_visuals/'.")