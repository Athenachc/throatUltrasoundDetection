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

# --- 1. SETUP MODELS ---
print("Initializing Models...")
try:
    GlobalHydra.instance().clear()
    initialize(config_path=".", version_base=None)

    yolo_path = "./runs/detect/train2/weights/best.pt"
    sam2_checkpoint = "/home/athena/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg_name = "sam2.1_hiera_l"

    yolo_model = YOLO(yolo_path)
    sam2_model = build_sam2(model_cfg_name, sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
except Exception as e:
    print(f"Initialization Error: {e}")
    sys.exit()

# --- 2. DYNAMIC PATH HANDLING ---
video_input = "/home/athena/Ultrasound_videos/Ultrasound_Scanning_Supplementary_Video/crop/Prior_Scan_zoom_v2.mp4"
if not os.path.exists(video_input):
    print(f"Error: Video file not found at {video_input}")
    sys.exit()

base_name = os.path.splitext(os.path.basename(video_input))[0]
output_video_path = f"{base_name}_roi.mp4"
csv_file = f"{base_name}_roi.csv"

# Naming Convention: TC, CC, T1, T2, T3, T4, T5, T6, T7, T8
BOUNDARY_NAMES = ["TC", "CC"] + [f"T{i}" for i in range(1, 9)]

# --- 3. LOGGING FUNCTION ---
def log_to_csv(frame_idx, total_roi, roi_dict):
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["Frame Index", "Total ROI (px)"] + BOUNDARY_NAMES
            writer.writerow(header)
        
        row = [frame_idx, total_roi]
        for name in BOUNDARY_NAMES:
            row.append(roi_dict.get(name, 0))
        writer.writerow(row)

# --- 4. VIDEO PROCESSING ---
cap = cv2.VideoCapture(video_input)
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = int(cap.get(3)), int(cap.get(4))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print(f"Processing: {video_input}")

frame_count = 0
while cap.isOpened():
    ret, frame_bgr = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    # A. YOLO Detection
    results = yolo_model.predict(frame_bgr, conf=0.5, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    
    roi_dict = {}
    total_frame_roi = 0
    
    if len(boxes) > 0:
        # # B. Sort Right-to-Left (x2 coordinate descending)
        # sorted_indices = np.argsort(boxes[:, 2])[::-1] 
        # sorted_boxes = boxes[sorted_indices]

        # B. SORT LEFT-TO-RIGHT
        # Sort by x1 (left edge) in ASCENDING order.
        # Smallest x values (left side) come first (index 0 = TC).
        sorted_indices = np.argsort(boxes[:, 0]) 
        sorted_boxes = boxes[sorted_indices]

        # C. SAM2 Prediction
        predictor.set_image(image_rgb)
        masks, _, _ = predictor.predict(box=sorted_boxes, multimask_output=False)
        
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        for i, mask in enumerate(masks):
            if i >= len(BOUNDARY_NAMES): break
            name = BOUNDARY_NAMES[i]
            
            # D. STRICT BOX CONSTRAINT
            x1, y1, x2, y2 = sorted_boxes[i].astype(int)
            box_constraint = np.zeros(mask.shape, dtype=bool)
            box_constraint[y1:y2, x1:x2] = True
            constrained_mask = np.logical_and(mask > 0, box_constraint)
            mask_uint8 = constrained_mask.astype(np.uint8) * 255
            
            # E. CLEANING & LABELING
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_cnt = max(contours, key=cv2.contourArea)
                area = int(cv2.contourArea(largest_cnt))
                
                if area > 100:
                    roi_dict[name] = area
                    total_frame_roi += area
                    
                    # 1. Draw Boundary
                    cv2.drawContours(frame_bgr, [largest_cnt], -1, (0, 255, 0), 2)
                    
                    # 2. Add Label Below Boundary
                    M = cv2.moments(largest_cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        # Place text slightly below the bottom edge of the contour
                        # Find the lowest point of the contour (max y)
                        cy_bottom = int(np.max(largest_cnt[:, :, 1]))
                        
                        # Draw name centered horizontally, just below the boundary
                        cv2.putText(frame_bgr, name, (cx - 20, cy_bottom + 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # F. Save results
    log_to_csv(frame_count, total_frame_roi, roi_dict)
    out.write(frame_bgr)
    
    frame_count += 1
    if frame_count % 20 == 0:
        print(f"Progress: {frame_count}/{total_frames} frames...", end="\r")

cap.release()
out.release()
print(f"\nDone! Analysis saved to {output_video_path}")