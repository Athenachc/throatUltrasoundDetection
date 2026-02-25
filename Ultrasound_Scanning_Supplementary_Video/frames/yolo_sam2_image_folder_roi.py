import cv2
import torch
import numpy as np
import os
import csv
import glob  # Added for folder scanning
from ultralytics import YOLO
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- 1. SETUP MODELS ---
GlobalHydra.instance().clear()
initialize(config_path=".", version_base=None)

yolo_path = "runs/detect/train5/weights/best.pt"
sam2_checkpoint = "/home/athena/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg_name = "sam2.1_hiera_l"

yolo_model = YOLO(yolo_path)
sam2_model = build_sam2(model_cfg_name, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)

# Paths
# input_folder = "/home/athena/Ultrasound_videos/human_test_results/Process_Data/Data_P_Human_Throat_US/cgl/filtered/20250930122939/"
input_folder = "/home/athena/Ultrasound_videos/Ultrasound_Scanning_Supplementary_Video/zoom/Smooth_Scan_zoom.mp4/"
output_folder = "processed_results_phantom"
csv_file = "ultrasound_shadow_analysis_phantom.csv"

os.makedirs(output_folder, exist_ok=True)

# --- 2. LOGGING UTILITY ---
def log_to_csv(img_name, count, total_roi, roi_list):
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            # Flexible header: allows for up to 20 individual shadow columns
            header = ["Image Name", "Shadow Count", "Total ROI (px)"] + [f"Shadow_{i+1}_ROI" for i in range(20)]
            writer.writerow(header)
        
        row = [img_name, count, total_roi] + roi_list
        writer.writerow(row)

# --- 3. PROCESSING LOGIC ---
def process_full_roi_analysis(image_path):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return
        
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_name = os.path.basename(image_path)
    
    results = yolo_model.predict(image_bgr, conf=0.5, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    
    if len(boxes) == 0:
        log_to_csv(image_name, 0, 0, [])
        cv2.imwrite(os.path.join(output_folder, f"res_{image_name}"), image_bgr)
        return

    predictor.set_image(image_rgb)
    masks, _, _ = predictor.predict(box=boxes, multimask_output=False)
    
    if masks.ndim == 4:
        masks = masks.squeeze(1) 

    individual_rois = []
    
    for i, mask in enumerate(masks):
        roi_area = int(np.sum(mask > 0))
        individual_rois.append(roi_area)
        
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            cv2.drawContours(image_bgr, contours, -1, (0, 255, 0), 2)
            M = cv2.moments(contours[0])
            if M["m00"] != 0:
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                cv2.putText(image_bgr, f"S{i+1}: {roi_area}", (cx, cy-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    total_roi = sum(individual_rois)
    log_to_csv(image_name, len(individual_rois), total_roi, individual_rois)
    
    # Save the processed image to the output folder
    cv2.imwrite(os.path.join(output_folder, f"res_{image_name}"), image_bgr)
    print(f"Processed: {image_name} | Shadows: {len(individual_rois)}")

# --- 4. EXECUTION LOOP ---
# Get all image files from the folder
image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(input_folder, ext)))

print(f"Found {len(image_files)} images. Starting batch processing...")

for img_path in sorted(image_files):
    process_full_roi_analysis(img_path)

print(f"\nProcessing complete. Data saved to {csv_file} and images to {output_folder}/")