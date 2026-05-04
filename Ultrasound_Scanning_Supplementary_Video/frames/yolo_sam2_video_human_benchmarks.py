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

# Fixed Colors for Anatomical Consistency (RGBA)
COLOR_MAP = {
    "TC": [0, 1, 0, 0.5],    # Green
    "CC": [1, 0, 0, 0.5],    # Red
    "T":  [0, 0, 1, 0.5]     # Blue (for T-rings)
}

# --- HELPER FUNCTIONS ---
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)

def save_visual_comparison(frame_rgb, gt_mask, model_masks_dict, model_labels_dict, video_name, frame_idx):
    full_data_models = ['train9', 'train10', 'train11']
    limited_data_models = ['train3', 'train5', 'train8']
    
    fig = plt.figure(figsize=(22, 10))
    gs = fig.add_gridspec(2, 5)

    # 1. Original
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(frame_rgb)
    ax_orig.set_title(f"Original\n({video_name} @ {frame_idx})")
    ax_orig.axis('off')

    # 2. Ground Truth (B&W + Cyan Labels)
    ax_gt = fig.add_subplot(gs[0, 1])
    bw_gt = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
    bw_gt[gt_mask > 0] = [255, 255, 255] 
    ax_gt.imshow(bw_gt)
    
    # Process GT labels
    num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(gt_mask)
    if num_labels > 1:
        # Sort components by area to find TC (usually the largest)
        # components are [background, comp1, comp2...]
        comp_indices = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1 
        
        tc_idx = comp_indices[0] # Largest is TC
        other_indices = comp_indices[1:]
        
        # Sort "others" by x-coordinate for CC/T ordering
        if len(other_indices) > 0:
            other_centroids = centroids[other_indices]
            x_sorted_sub_indices = np.argsort(other_centroids[:, 0])
            other_indices = other_indices[x_sorted_sub_indices]

        # Draw TC Label
        ax_gt.text(centroids[tc_idx][0], centroids[tc_idx][1] + 60, "TC", color='cyan', 
                   fontsize=10, fontweight='bold', ha='center', 
                   bbox=dict(facecolor='black', alpha=0.8, pad=1))
        
        # Draw CC and T labels
        t_count = 1
        for i, idx in enumerate(other_indices):
            cx, cy = centroids[idx]
            gt_lbl = "CC" if i == 0 else f"T{t_count}"
            if i > 0: t_count += 1
            ax_gt.text(cx, cy + 35, gt_lbl, color='cyan', fontsize=10, fontweight='bold', 
                       ha='center', bbox=dict(facecolor='black', alpha=0.8, pad=1))
            
    ax_gt.set_title("Ground Truth (B&W)")
    ax_gt.axis('off')

    # 3. Model Panels
    def plot_with_labels(ax, m_id, title):
        ax.imshow(frame_rgb)
        if m_id in model_masks_dict and model_masks_dict[m_id] is not None:
            ax.imshow(model_masks_dict[m_id])
            if m_id in model_labels_dict:
                for label, coords in model_labels_dict[m_id]:
                    ax.text(coords[0], coords[1] + 35, label, color='white', fontsize=11, 
                            fontweight='bold', ha='center',
                            bbox=dict(facecolor='black', alpha=0.9, pad=0.5))
        ax.set_title(title)
        ax.axis('off')

    for i, m_id in enumerate(full_data_models):
        plot_with_labels(fig.add_subplot(gs[0, i+2]), m_id, f"{m_id} (Full)")
    for i, m_id in enumerate(limited_data_models):
        plot_with_labels(fig.add_subplot(gs[1, i+1]), m_id, f"{m_id} (Limited)")

    plt.tight_layout()
    os.makedirs("validation_visuals", exist_ok=True)
    plt.savefig(f"validation_visuals/{video_name}_F{frame_idx:05d}.png", bbox_inches='tight', dpi=150)
    plt.close()

# --- CORE BENCHMARK ENGINE ---
def run_hybrid_benchmark(yolo_paths, sam2_predictor, video_path, gt_folder):
    video_name = os.path.basename(video_path)
    v_base_name = os.path.splitext(video_name)[0]
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    all_gt_paths = glob(os.path.join(gt_folder, f"{v_base_name}_*.png"))
    target_frames = {int(re.search(r'_(\d+)\.png$', p).group(1)): p for p in all_gt_paths if re.search(r'_(\d+)\.png$', p)}
    
    accumulated_metrics = {os.path.basename(os.path.dirname(os.path.dirname(p))): {"latencies":[], "dice":[], "tp":[]} for p in yolo_paths}

    for frame_idx in sorted(target_frames.keys()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        gt_mask_raw = cv2.imread(target_frames[frame_idx], 0)
        gt_mask = np.where(gt_mask_raw > 0, 1, 0).astype(np.uint8)
        coords_gt = cv2.findNonZero(gt_mask_raw)
        gt_box = None
        if coords_gt is not None:
            gx, gy, gw, gh = cv2.boundingRect(coords_gt)
            gt_box = [gx, gy, gx + gw, gy + gh]

        frame_masks = {}
        frame_labels = {}

        for yolo_p in yolo_paths:
            m_id = yolo_p.split('/')[-3]
            model = YOLO(yolo_p).to("cuda")
            
            start_time = time.perf_counter()
            results = model.predict(frame, conf=0.5, verbose=False)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            
            vis_mask = np.zeros((*frame.shape[:2], 4))
            bin_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            lbl_coords = []
            is_tp = 0

            if len(boxes) > 0:
                tc_indices = [i for i, cid in enumerate(class_ids) if cid in [0, 2]]
                cart_indices = [i for i, cid in enumerate(class_ids) if cid == 1]
                
                sam2_predictor.set_image(frame_rgb)
                
                # REVISED LOGIC: Only 1 TC allowed, CC only if TC exists
                if tc_indices:
                    # Pick the TC closest to center/edges based on prior logic
                    cx_vals = (boxes[tc_indices, 0] + boxes[tc_indices, 2]) / 2
                    best_tc_idx = tc_indices[np.argmin(np.minimum(cx_vals, width - cx_vals))]
                    tc_side = 'left' if cx_vals[np.argmin(np.minimum(cx_vals, width - cx_vals))] < width/2 else 'right'
                    
                    # Segment TC
                    m, _, _ = sam2_predictor.predict(box=boxes[best_tc_idx], multimask_output=False)
                    tc_m = m[0].astype(np.uint8)
                    bin_mask = np.maximum(bin_mask, tc_m)
                    vis_mask[tc_m > 0] = COLOR_MAP["TC"]
                    lbl_coords.append(("TC", (int((boxes[best_tc_idx][0]+boxes[best_tc_idx][2])/2), int(boxes[best_tc_idx][3]))))
                    
                    if gt_box and calculate_iou(boxes[best_tc_idx], gt_box) >= 0.5: is_tp = 1

                    # If TC exists, the first cartilage is CC
                    if cart_indices:
                        cc_x = (boxes[cart_indices, 0] + boxes[cart_indices, 2]) / 2
                        s_idx = np.argsort(cc_x)[::-1] if tc_side == 'right' else np.argsort(cc_x)
                        
                        t_cnt = 1
                        for i, idx in enumerate(s_idx):
                            act_idx = cart_indices[idx]
                            m, _, _ = sam2_predictor.predict(box=boxes[act_idx], multimask_output=False)
                            inst_m = m[0].astype(np.uint8)
                            bin_mask = np.maximum(bin_mask, inst_m)
                            
                            if i == 0:
                                vis_mask[inst_m > 0] = COLOR_MAP["CC"]
                                lbl_coords.append(("CC", (int((boxes[act_idx][0]+boxes[act_idx][2])/2), int(boxes[act_idx][3]))))
                            else:
                                vis_mask[inst_m > 0] = COLOR_MAP["T"]
                                lbl_coords.append((f"T{t_cnt}", (int((boxes[act_idx][0]+boxes[act_idx][2])/2), int(boxes[act_idx][3]))))
                                t_cnt += 1
                else:
                    # NO TC: All cartilages are T1, T2...
                    if cart_indices:
                        cc_x = (boxes[cart_indices, 0] + boxes[cart_indices, 2]) / 2
                        s_idx = np.argsort(cc_x) # Default left-to-right
                        for i, idx in enumerate(s_idx):
                            act_idx = cart_indices[idx]
                            m, _, _ = sam2_predictor.predict(box=boxes[act_idx], multimask_output=False)
                            inst_m = m[0].astype(np.uint8)
                            bin_mask = np.maximum(bin_mask, inst_m)
                            vis_mask[inst_m > 0] = COLOR_MAP["T"]
                            lbl_coords.append((f"T{i+1}", (int((boxes[act_idx][0]+boxes[act_idx][2])/2), int(boxes[act_idx][3]))))

            accumulated_metrics[m_id]["latencies"].append((time.perf_counter() - start_time)*1000)
            accumulated_metrics[m_id]["dice"].append(dc(bin_mask, gt_mask))
            accumulated_metrics[m_id]["tp"].append(is_tp)
            frame_masks[m_id] = vis_mask
            frame_labels[m_id] = lbl_coords
            del model
            torch.cuda.empty_cache()

        save_visual_comparison(frame_rgb, gt_mask, frame_masks, frame_labels, v_base_name, frame_idx)

    cap.release()
    return [{ "Video": video_name, "Model": m, 
              "FPS": round(1000/np.mean(d["latencies"]), 1) if d["latencies"] else 0,
              "Mean Dice": round(np.mean(d["dice"]), 4) if d["dice"] else 0,
              "mAP@50": round(np.mean(d["tp"]), 4) if d["tp"] else 0 
            } for m, d in accumulated_metrics.items()]

if __name__ == "__main__":
    GlobalHydra.instance().clear()
    initialize(config_path=".", version_base=None)
    sam2_predictor = SAM2ImagePredictor(build_sam2("sam2.1_hiera_l", "/home/athena/sam2/checkpoints/sam2.1_hiera_large.pt", device="cuda"))

    all_results = []
    for vp in sorted(glob(os.path.join(video_folder, "*.mp4"))):
        all_results.extend(run_hybrid_benchmark(yolo_models_list, sam2_predictor, vp, gt_folder))

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv("hybrid_benchmark_results.csv", index=False)
        print("\nSUMMARY:\n", df.groupby('Model').mean(numeric_only=True))