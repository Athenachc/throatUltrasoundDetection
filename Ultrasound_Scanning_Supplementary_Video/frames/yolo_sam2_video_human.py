import cv2
import torch
import numpy as np
import os
import csv
import sys
from ultralytics import YOLO
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- 0. PRE-FLIGHT ---
torch.cuda.empty_cache()
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# --- 1. SETUP MODELS ---
print("Initializing Models...")
try:
    GlobalHydra.instance().clear()
    initialize(config_path=".", version_base=None)
    # train2 for phantom data, train3 for human data (YOLOv8 nano)
    # train6 for phantom data,train5 for human data (yolo11n)
    # train7 for phantom data,train8 for human data (yolov10n)
    # train9 for online_human data (yolov8n)
    # train10 from online_human data (yolov10n)
    # train11 from online_human data (yolo11n)
    yolo_path = "./runs/detect/train9/weights/best.pt" 
    sam2_checkpoint = "/home/athena/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg_name = "sam2.1_hiera_l"

    yolo_model = YOLO(yolo_path)
    sam2_model = build_sam2(model_cfg_name, sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
except Exception as e:
    print(f"Initialization Error: {e}")
    sys.exit()

# --- 2. PATHS & VIDEO CONFIG ---
video_input = "/home/athena/Ultrasound_videos/online_videos/video7.mp4"
if not os.path.exists(video_input):
    print(f"Error: Video not found at {video_input}")
    sys.exit()

base_name = os.path.splitext(os.path.basename(video_input))[0]
output_video_path = f"{base_name}_roi.mp4"
csv_file = f"{base_name}_roi.csv"
BOUNDARY_NAMES_ALL = ["TC", "CC"] + [f"T{i}" for i in range(1, 15)]

# --- 3. LOGGING FUNCTION ---
def log_to_csv(frame_idx, total_roi, roi_dict):
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["Frame Index", "Total ROI (px)"] + BOUNDARY_NAMES_ALL
            writer.writerow(header)
        row = [frame_idx, total_roi]
        for name in BOUNDARY_NAMES_ALL:
            row.append(roi_dict.get(name, 0))
        writer.writerow(row)

# --- 4. VIDEO PROCESSING ---
cap = cv2.VideoCapture(video_input)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print(f"Processing: {video_input}")

frame_count = 0
with torch.inference_mode():
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret: break

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = yolo_model.predict(frame_bgr, conf=0.5, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        roi_dict = {}
        total_frame_roi = 0
        
        if len(boxes) > 0:
            final_names = [None] * len(boxes)
            
            # Step 1: Find all TC candidates (Class 0 or 2)
            tc_indices = [i for i, cid in enumerate(class_ids) if cid in [0, 2]]
            cart_indices = [i for i, cid in enumerate(class_ids) if cid == 1]
            
            tc_side = None # 'left' or 'right'
            has_tc = False

            if tc_indices:
                # Calculate centers
                centers_x = (boxes[tc_indices, 0] + boxes[tc_indices, 2]) / 2
                
                # Distance to left (x=0) and right (x=width)
                dist_to_left = centers_x 
                dist_to_right = width - centers_x
                
                # Find the single index that is closest to ANY edge
                min_left_idx = np.argmin(dist_to_left)
                min_right_idx = np.argmin(dist_to_right)
                
                if dist_to_left[min_left_idx] < dist_to_right[min_right_idx]:
                    best_tc_idx = tc_indices[min_left_idx]
                    tc_side = 'left'
                else:
                    best_tc_idx = tc_indices[min_right_idx]
                    tc_side = 'right'
                
                final_names[best_tc_idx] = "TC"
                has_tc = True

            # Step 2: Label Cartilages based on TC position
            if cart_indices:
                c_boxes = boxes[cart_indices]
                c_centers_x = (c_boxes[:, 0] + c_boxes[:, 2]) / 2
                
                # If TC is on right, CC is the rightmost cartilage. If TC is on left, CC is leftmost.
                # Sorting logic:
                if tc_side == 'right':
                    # Sort Cartilages Right to Left (Descending X)
                    c_sort_idx = np.argsort(c_centers_x)[::-1]
                else:
                    # Sort Cartilages Left to Right (Ascending X - Default for TC on left or no TC)
                    c_sort_idx = np.argsort(c_centers_x)
                
                t_counter = 1
                for i, idx in enumerate(c_sort_idx):
                    actual_idx = cart_indices[idx]
                    if has_tc and i == 0:
                        final_names[actual_idx] = "CC"
                    else:
                        final_names[actual_idx] = f"T{t_counter}"
                        t_counter += 1

            # Step 3: SAM2 Segmentation
            predictor.set_image(image_rgb)
            masks, _, _ = predictor.predict(box=boxes, multimask_output=False)
            if masks.ndim == 4: masks = masks.squeeze(1)

            for i, mask in enumerate(masks):
                name = final_names[i]
                if name is None: continue 

                mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
                x1, y1, x2, y2 = boxes[i].astype(int)
                
                box_mask = np.zeros(mask_np.shape, dtype=bool)
                box_mask[y1:y2, x1:x2] = True
                mask_uint8 = (np.logical_and(mask_np > 0, box_mask)).astype(np.uint8) * 255
                
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_cnt = max(contours, key=cv2.contourArea)
                    area = int(cv2.contourArea(largest_cnt))
                    if area > 100:
                        roi_dict[name] = area
                        total_frame_roi += area
                        
                        color = (0, 255, 0) if name == "TC" else (255, 0, 0)
                        cv2.drawContours(frame_bgr, [largest_cnt], -1, color, 2)
                        
                        M = cv2.moments(largest_cnt)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy_bottom = int(np.max(largest_cnt[:, :, 1]))
                            cv2.putText(frame_bgr, name, (cx - 20, cy_bottom + 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        log_to_csv(frame_count, total_frame_roi, roi_dict)
        out.write(frame_bgr)
        
        frame_count += 1
        if frame_count % 20 == 0:
            print(f"Progress: {frame_count}/{total_frames} frames...", end="\r")
            torch.cuda.empty_cache()

cap.release()
out.release()
print(f"\nDone! Video: {output_video_path} | CSV: {csv_file}")