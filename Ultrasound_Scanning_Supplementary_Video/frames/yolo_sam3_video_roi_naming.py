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
    # Optimizations for your RTX 5070 Ti
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# --- 1. SETUP MODELS ---
print("Initializing SAM 3 and YOLO (Phantom Tuning)...")
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    # SAM 3 via ModelScope
    model_path = snapshot_download('facebook/sam3')
    model = Sam3Model.from_pretrained(model_path).to(device)
    processor = Sam3Processor.from_pretrained(model_path)

    # YOLO Path updated to train2 (Phantom Weights)
    yolo_path = "./runs/detect/train2/weights/best.pt" 
    yolo_model = YOLO(yolo_path)
except Exception as e:
    print(f"Initialization Error: {e}")
    sys.exit()

# --- 2. DYNAMIC PATH HANDLING ---
video_input = "/home/athena/Ultrasound_videos/Ultrasound_Scanning_Supplementary_Video/crop/RL_PPO3_zoom_v2.mp4"
if not os.path.exists(video_input):
    print(f"Error: Video not found at {video_input}")
    sys.exit()

base_name = os.path.splitext(os.path.basename(video_input))[0]
output_video_path = f"{base_name}_sam3_phantom.mp4"
csv_file = f"{base_name}_sam3_phantom.csv"

# Fixed Pool for CSV columns
BOUNDARY_NAMES_ALL = ["TC", "CC"] + [f"T{i}" for i in range(1, 14)]

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
width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print(f"Processing Phantom with SAM 3: {video_input}")

frame_count = 0
with torch.no_grad():
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret: break

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = yolo_model.predict(frame_bgr, conf=0.5, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        
        roi_dict = {}
        total_frame_roi = 0
        
        if len(boxes) > 0:
            # Sort ALL boxes from Left-to-Right
            centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
            sorted_indices = np.argsort(centers_x)
            sorted_boxes = boxes[sorted_indices]
            sorted_centers_x = centers_x[sorted_indices]

            # Territory-based naming logic (Identical to your provided script)
            box_names = []
            t_counter = 1
            tc_assigned = False
            cc_assigned = False

            for cx in sorted_centers_x:
                if cx < 500 and not tc_assigned:
                    box_names.append("TC")
                    tc_assigned = True
                elif 500 <= cx < 800 and not cc_assigned:
                    box_names.append("CC")
                    cc_assigned = True
                else:
                    box_names.append(f"T{t_counter}")
                    t_counter += 1

            # --- TUNED SAM 3 SEGMENTATION ---
            inputs = processor(
                images=image_rgb, 
                input_boxes=[sorted_boxes.tolist()], 
                return_tensors="pt"
            ).to(device)

            # Explicitly label boxes as foreground for SAM 3
            inputs["input_labels"] = torch.ones((1, len(sorted_boxes)), dtype=torch.long, device=device)

            # Request multimask_output=True to catch best possible boundary
            outputs = model(**inputs, multimask_output=True)
            
            # Use 0.3 threshold (as you found 0.1/0.05 too noisy for shadows)
            masks_results = processor.post_process_instance_segmentation(
                outputs, threshold=0.3, target_sizes=[(height, width)]
            )[0]
            
            masks = masks_results["masks"]

            for i, mask in enumerate(masks):
                # Guardrail for index mismatch
                if i >= len(box_names): break
                name = box_names[i]
                
                # Multi-mask selection: Pick largest mask if SAM 3 returned options
                if mask.ndim == 3:
                    areas = [torch.sum(m).item() for m in mask]
                    mask_final_tensor = mask[np.argmax(areas)]
                else:
                    mask_final_tensor = mask

                mask_np = (mask_final_tensor.cpu().numpy() > 0).astype(np.uint8) * 255
                x1, y1, x2, y2 = sorted_boxes[i].astype(int)
                
                # Apply Spatial Constraint within YOLO box
                box_mask = np.zeros_like(mask_np)
                box_mask[max(0,y1):min(height,y2), max(0,x1):min(width,x2)] = 255
                mask_display = cv2.bitwise_and(mask_np, box_mask)
                
                contours, _ = cv2.findContours(mask_display, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_cnt = max(contours, key=cv2.contourArea)
                    area = int(cv2.contourArea(largest_cnt))
                    
                    if area > 100:
                        roi_dict[name] = area
                        total_frame_roi += area
                        
                        # Use Green boundary for visualization
                        cv2.drawContours(frame_bgr, [largest_cnt], -1, (0, 255, 0), 2)
                        
                        M = cv2.moments(largest_cnt)
                        if M["m00"] != 0:
                            cx_label = int(M["m10"] / M["m00"])
                            cy_bottom = int(np.max(largest_cnt[:, :, 1]))
                            cv2.putText(frame_bgr, name, (cx_label - 20, cy_bottom + 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        log_to_csv(frame_count, total_frame_roi, roi_dict)
        out.write(frame_bgr)
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Progress: {frame_count}/{total_frames} frames...", end="\r")

cap.release()
out.release()
print(f"\nDone! Video: {output_video_path} | CSV: {csv_file}")