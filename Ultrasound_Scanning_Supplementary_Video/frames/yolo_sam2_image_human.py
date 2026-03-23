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

    yolo_path = "./runs/detect/train3/weights/best.pt" # train3 for human, train2 for phantom
    sam2_checkpoint = "/home/athena/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg_name = "sam2.1_hiera_l"

    yolo_model = YOLO(yolo_path)
    sam2_model = build_sam2(model_cfg_name, sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)
except Exception as e:
    print(f"Initialization Error: {e}")
    sys.exit()

# --- 2. INPUT HANDLING ---
image_input = "/home/athena/Ultrasound_videos/human_test_results/data_collect-1215/20251215171729/2025_12_15_17_17_35_373.jpg" 
base_name = os.path.splitext(os.path.basename(image_input))[0]
output_image_path = f"{base_name}_result.jpg"
csv_file = f"{base_name}_roi.csv"

# Updated Pool for CSV columns
BOUNDARY_NAMES_ALL = ["TC", "CC"] + [f"T{i}" for i in range(1, 15)]

# --- 3. PROCESSING ---
frame_bgr = cv2.imread(image_input)
if frame_bgr is None:
    print("Error: Could not read image.")
    sys.exit()

image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
results = yolo_model.predict(frame_bgr, conf=0.5, verbose=False)[0]

# Extract boxes and their associated class IDs
# Class 0: Shadow, Class 1: Cartilage, Class 2: TC
boxes = results.boxes.xyxy.cpu().numpy()
classes = results.boxes.cls.cpu().numpy()

roi_dict = {}
total_roi = 0

if len(boxes) > 0:
    # Sort everything Left-to-Right
    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
    sorted_indices = np.argsort(centers_x)
    sorted_boxes = boxes[sorted_indices]
    sorted_classes = classes[sorted_indices]

    box_names = []
    
    # Check if TC (Class 2) exists anywhere in the detections
    has_tc = 2 in sorted_classes
    
    if has_tc:
        # Scenario A: TC detected. Order: TC -> CC -> T1 -> T2...
        tc_found = False
        cc_assigned = False
        t_counter = 1
        
        for i, cls_id in enumerate(sorted_classes):
            if cls_id == 2 and not tc_found:
                box_names.append("TC")
                tc_found = True
            elif tc_found and not cc_assigned:
                # The first cartilage to the right of TC is CC
                box_names.append("CC")
                cc_assigned = True
            else:
                # Everything else is T-series
                box_names.append(f"T{t_counter}")
                t_counter += 1
    else:
        # Scenario B: No TC. Order: T1 -> T2 -> T3...
        t_counter = 1
        for i in range(len(sorted_boxes)):
            box_names.append(f"T{t_counter}")
            t_counter += 1

    # SAM2 Prediction
    predictor.set_image(image_rgb)
    masks, _, _ = predictor.predict(box=sorted_boxes, multimask_output=False)
    if masks.ndim == 4: masks = masks.squeeze(1)

    for i, mask in enumerate(masks):
        name = box_names[i]
        x1, y1, x2, y2 = sorted_boxes[i].astype(int)
        
        box_constraint = np.zeros(mask.shape, dtype=bool)
        box_constraint[y1:y2, x1:x2] = True
        constrained_mask = np.logical_and(mask > 0, box_constraint)
        mask_uint8 = constrained_mask.astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_cnt = max(contours, key=cv2.contourArea)
            area = int(cv2.contourArea(largest_cnt))
            
            if area > 100:
                roi_dict[name] = area
                total_roi += area
                cv2.drawContours(frame_bgr, [largest_cnt], -1, (0, 255, 0), 2)
                
                M = cv2.moments(largest_cnt)
                if M["m00"] != 0:
                    cx_label = int(M["m10"] / M["m00"])
                    cy_bottom = int(np.max(largest_cnt[:, :, 1]))
                    cv2.putText(frame_bgr, name, (cx_label - 20, cy_bottom + 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# --- 4. SAVE RESULTS ---
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    header = ["Image Name", "Total ROI (px)"] + BOUNDARY_NAMES_ALL
    writer.writerow(header)
    row = [image_input, total_roi]
    for name in BOUNDARY_NAMES_ALL:
        row.append(roi_dict.get(name, 0))
    writer.writerow(row)

cv2.imwrite(output_image_path, frame_bgr)
print(f"Done! Result saved to {output_image_path}")