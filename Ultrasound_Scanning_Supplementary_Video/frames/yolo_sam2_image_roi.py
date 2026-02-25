import cv2
import torch
import numpy as np
import os
import csv
from ultralytics import YOLO
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- 1. SETUP MODELS ---
GlobalHydra.instance().clear()
initialize(config_path=".", version_base=None)

yolo_path = "runs/detect/train4/weights/best.pt"
sam2_checkpoint = "/home/athena/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg_name = "sam2.1_hiera_l"

yolo_model = YOLO(yolo_path)
sam2_model = build_sam2(model_cfg_name, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)

# Result CSV Path
csv_file = "ultrasound_shadow_analysis.csv"

# --- 2. PROCESSING & LOGGING ---
def process_full_roi_analysis(image_path):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return None
        
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_name = os.path.basename(image_path)
    
    # YOLO Detection
    results = yolo_model.predict(image_bgr, conf=0.5, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    
    if len(boxes) == 0:
        log_to_csv(image_name, 0, 0, [])
        return image_bgr

    # SAM2 Prediction
    predictor.set_image(image_rgb)
    masks, _, _ = predictor.predict(box=boxes, multimask_output=False)
    
    if masks.ndim == 4:
        masks = masks.squeeze(1) 

    individual_rois = []
    
    # Processing each shadow
    for i, mask in enumerate(masks):
        # Calculate ROI
        roi_area = int(np.sum(mask > 0))
        individual_rois.append(roi_area)
        
        # Visualization
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image_bgr, contours, -1, (0, 255, 0), 2)
        
        # Add Label and Individual ROI to the image
        M = cv2.moments(contours[0])
        if M["m00"] != 0:
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            cv2.putText(image_bgr, f"S{i+1}: {roi_area}", (cx, cy-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    total_roi = sum(individual_rois)
    
    # Log to CSV
    log_to_csv(image_name, len(individual_rois), total_roi, individual_rois)
    
    return image_bgr

def log_to_csv(img_name, count, total_roi, roi_list):
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            # Header layout
            header = ["Image Name", "Shadow Count", "Total ROI (px)"] + [f"Shadow_{i+1}_ROI" for i in range(10)]
            writer.writerow(header)
        
        # Data row: Image info + Total + Individual values
        row = [img_name, count, total_roi] + roi_list
        writer.writerow(row)
    print(f"{img_name} logged: Total ROI = {total_roi}px across {count} shadows.")

# --- 3. EXECUTION ---
test_frame = "/home/athena/Ultrasound_videos/Ultrasound_Scanning_Supplementary_Video/frames/dataset_human/train/images/2025_09_30_12_36_55_586.jpg"

result_img = process_full_roi_analysis(test_frame)

if result_img is not None:
    cv2.imwrite("final_roi_analysis_result.jpg", result_img)
    cv2.imshow("Complete ROI Analysis", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()