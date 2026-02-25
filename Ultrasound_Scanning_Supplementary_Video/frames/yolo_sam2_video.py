import cv2
import torch
import numpy as np
import os
from ultralytics import YOLO
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- 1. CONFIGURATION ---
# Define BASE_DIR first so the rest of the script can see it!
BASE_DIR = "/home/athena/Ultrasound_videos/Ultrasound_Scanning_Supplementary_Video/frames"
VIDEO_INPUT = "/home/athena/Ultrasound_videos/Ultrasound_Scanning_Supplementary_Video/crop/Smooth_Scan_zoom_v3.mp4"

# Dynamic Naming Logic:
# This takes "xxx.mp4" and turns it into "xxx_boundary.mp4"
base_name = os.path.splitext(os.path.basename(VIDEO_INPUT))[0] # Gets "xxx"
VIDEO_OUTPUT = f"{base_name}_boundary.mp4"

# Setup Hydra for SAM2
GlobalHydra.instance().clear()
initialize(config_path=".", version_base=None)

# Paths
yolo_path = os.path.join(BASE_DIR, "runs/detect/train/weights/best.pt")
sam2_checkpoint = "/home/athena/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg_name = "sam2.1_hiera_l"

# Load Models
print("Loading YOLO and SAM2.1...")
yolo_model = YOLO(yolo_path)
sam2_model = build_sam2(model_cfg_name, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)

# --- 2. VIDEO SETUP ---
cap = cv2.VideoCapture(VIDEO_INPUT)
if not cap.isOpened():
    print(f"Error: Could not open video {VIDEO_INPUT}")
    exit()

# Get video properties for saving
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, fps, (width, height))

print(f"🎬 Processing video: {width}x{height} @ {fps} FPS")

# --- 3. THE LOOP ---
frame_count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break 

    frame_count += 1
    # Print progress every 10 frames
    if frame_count % 10 == 0:
        print(f"Processing frame {frame_count}/{total_frames}...", end="\r")

    # A. YOLO Detection
    results = yolo_model.predict(frame, conf=0.5, verbose=False, device="cuda")[0]
    boxes = results.boxes.xyxy.cpu().numpy()

    if len(boxes) > 0:
        # B. SAM2 Segmentation
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)
        
        # We only need to call this ONCE
        masks, scores, _ = predictor.predict(box=boxes, multimask_output=False)
        
        # Robust Squeeze: Convert (N, 1, H, W) to (N, H, W)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)
        elif masks.ndim == 3:
            # If SAM2 returns (1, H, W) for a single box, it's already good
            pass

        # C. Visualization (Draw on frame)
        for mask in masks:
            # Create a boolean mask for faster indexing
            mask_bool = mask > 0.0
            mask_uint8 = (mask_bool).astype(np.uint8) * 255
            
            # Find and draw the green boundary
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
            
            # Apply semi-transparent green fill
            frame[mask_bool] = frame[mask_bool] * 0.7 + np.array([0, 100, 0]) * 0.3

    # D. Save to File and Display
    out.write(frame) # This line saves the frame to your MP4 file
    cv2.imshow("Ultrasound Live Segmentation", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nProcessing stopped early by user.")
        break

# Cleanup
cap.release()
out.release() # This "closes" the video file so it can be played
cv2.destroyAllWindows()
print(f"\n Finished! Video saved as: {VIDEO_OUTPUT}")