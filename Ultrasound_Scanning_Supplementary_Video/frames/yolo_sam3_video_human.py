import cv2
import torch
import numpy as np
import os
import csv
import sys
from ultralytics import YOLO
from transformers import Sam3Processor, Sam3Model
from modelscope import snapshot_download

# --- 0. PRE-FLIGHT ---
torch.cuda.empty_cache()
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# --- 1. SETUP MODELS ---
print("Initializing SAM 3 (Tuned) and YOLO...")
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    model_path = snapshot_download('facebook/sam3')
    model = Sam3Model.from_pretrained(model_path).to(device)
    processor = Sam3Processor.from_pretrained(model_path)
    yolo_model = YOLO("./runs/detect/train3/weights/best.pt")
except Exception as e:
    print(f"Initialization Error: {e}")
    sys.exit()

# --- 2. PATHS & VIDEO CONFIG ---
video_input = "/home/athena/Ultrasound_videos/human_test_results/Process_Data/Data_P_Human_Throat_US/ywc/filtered/20250930121841/20250930121841_shortlisted_v2.mp4"
if not os.path.exists(video_input):
    print(f"Error: Video not found at {video_input}")
    sys.exit()

base_name = os.path.splitext(os.path.basename(video_input))[0]
output_video_path = f"{base_name}_sam3_tuned.mp4"
csv_file = f"{base_name}_sam3_tuned.csv"
BOUNDARY_NAMES_ALL = ["TC", "CC"] + [f"T{i}" for i in range(1, 15)]

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
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_count = 0
with torch.no_grad():
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret: break

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = yolo_model.predict(frame_bgr, conf=0.5, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        roi_dict, total_frame_roi = {}, 0
        
        if len(boxes) > 0:
            # Step 1: Naming Logic
            tc_indices = [i for i, cid in enumerate(class_ids) if cid in [0, 2] and (boxes[i][0] + boxes[i][2])/2 < 500]
            cart_indices = [i for i, cid in enumerate(class_ids) if cid == 1]
            final_names = [None] * len(boxes)
            
            has_tc = False
            if tc_indices:
                tc_idx = tc_indices[np.argmin((boxes[tc_indices, 0] + boxes[tc_indices, 2]) / 2)]
                final_names[tc_idx] = "TC"
                has_tc = True
            if cart_indices:
                c_sort_idx = np.argsort((boxes[cart_indices, 0] + boxes[cart_indices, 2]) / 2)
                t_counter = 1
                for i, idx in enumerate(c_sort_idx):
                    actual_idx = cart_indices[idx]
                    if has_tc and i == 0: final_names[actual_idx] = "CC"
                    else:
                        final_names[actual_idx] = f"T{t_counter}"
                        t_counter += 1

            # Step 2: TUNED SAM 3 Segmentation
            inputs = processor(images=image_rgb, input_boxes=[boxes.tolist()], return_tensors="pt").to(device)
            inputs["input_labels"] = torch.ones((1, len(boxes)), dtype=torch.long, device=device)
            
            # Using multimask_output=True to get more candidate boundaries
            outputs = model(**inputs, multimask_output=True)
            
            # Use 0.3 as requested for better shadow stability
            masks_results = processor.post_process_instance_segmentation(
                outputs, threshold=0.3, target_sizes=[(height, width)]
            )[0]
            
            masks = masks_results["masks"] # Shape: [num_boxes, 3, H, W] if multimask, or flattened

            # Step 3: Draw and Log
            for i, mask in enumerate(masks):
                if i >= len(final_names): break 
                name = final_names[i]
                if name is None: continue 

                # Handle multi-mask output (pick the largest valid mask per box)
                if mask.ndim == 3: # If SAM3 returns [3, H, W] for one box
                    areas = [torch.sum(m).item() for m in mask]
                    best_m_idx = np.argmax(areas)
                    mask_final_tensor = mask[best_m_idx]
                else:
                    mask_final_tensor = mask

                mask_np = (mask_final_tensor.cpu().numpy() > 0).astype(np.uint8) * 255
                
                # Spatial constraint to YOLO box
                x1, y1, x2, y2 = boxes[i].astype(int)
                box_mask = np.zeros_like(mask_np)
                box_mask[max(0,y1):min(height,y2), max(0,x1):min(width,x2)] = 255
                mask_display = cv2.bitwise_and(mask_np, box_mask)
                
                contours, _ = cv2.findContours(mask_display, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_cnt = max(contours, key=cv2.contourArea)
                    area = int(cv2.contourArea(largest_cnt))
                    if area > 100:
                        roi_dict[name], total_frame_roi = area, total_frame_roi + area
                        color = (0, 255, 0) if name == "TC" else (255, 0, 0)
                        cv2.drawContours(frame_bgr, [largest_cnt], -1, color, 2)
                        M = cv2.moments(largest_cnt)
                        if M["m00"] != 0:
                            cx, cy_bottom = int(M["m10"] / M["m00"]), int(np.max(largest_cnt[:, :, 1]))
                            cv2.putText(frame_bgr, name, (cx-20, cy_bottom+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        log_to_csv(frame_count, total_frame_roi, roi_dict)
        out.write(frame_bgr)
        frame_count += 1
        if frame_count % 10 == 0: print(f"Progress: {frame_count}/{total_frames} frames...", end="\r")

cap.release()
out.release()
print(f"\nDone! Tuned Video: {output_video_path}")