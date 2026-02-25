import cv2
import torch
import numpy as np
from ultralytics import YOLO
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- 1. SETUP MODELS ---
GlobalHydra.instance().clear()
initialize(config_path=".", version_base=None)

# Paths (Adjust if necessary)
# yolo_path = "runs/detect/train2/weights/best.pt"
yolo_path = "runs/detect/train4/weights/best.pt"
sam2_checkpoint = "/home/athena/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg_name = "sam2.1_hiera_l"

# Load YOLO
yolo_model = YOLO(yolo_path)

# Load SAM2
sam2_model = build_sam2(model_cfg_name, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)

# --- 2. THE SEGMENTATION FUNCTION ---
def segment_shadows(image_path):
    # Load image
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Step A: YOLO Detection
    results = yolo_model.predict(image_bgr, conf=0.5, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2] format
    
    if len(boxes) == 0:
        print("No shadows detected by YOLO.")
        return image_bgr

    # Step B: SAM2 Prediction
    predictor.set_image(image_rgb)
    
    # Pass all YOLO boxes at once as prompts
    masks, scores, _ = predictor.predict(
        box=boxes,
        multimask_output=False
    )
    
    # Step C: Visualization
    # masks is [N, 1, H, W], we convert to [N, H, W]
    masks = masks.squeeze(1) 
    
    for i, mask in enumerate(masks):
        # Create a green overlay for the shadow
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw the boundary (the "dome" edge)
        cv2.drawContours(image_bgr, contours, -1, (0, 255, 0), 2)
        
        # Optional: Add a semi-transparent fill
        image_bgr[mask > 0] = image_bgr[mask > 0] * 0.7 + np.array([0, 100, 0]) * 0.3

    return image_bgr

# --- 3. RUN AND SAVE ---
test_frame = "/home/athena/Ultrasound_videos/human_test_results/0930-cuhk-002/cmc/20250930122744/2025_09_30_12_27_51_715.jpg"
# test_frame = "/home/athena/Ultrasound_videos/Ultrasound_Scanning_Supplementary_Video/frames/dataset_human/train/images/2025_09_30_12_36_55_586.jpg"


result_img = segment_shadows(test_frame)

if result_img is not None:
    # 1. Save the image to your disk
    output_filename = "shadow_result_000001.jpg"
    cv2.imwrite(output_filename, result_img)
    print(f"Image successfully saved as: {output_filename}")

    # 2. Show the image window
    cv2.imshow("YOLO + SAM2 Ultrasound Result", result_img)
    print("Press any key on the image window to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()