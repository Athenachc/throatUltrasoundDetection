from ultralytics import YOLO
import cv2

# 1. Load your best trained model
#model = YOLO("./runs/detect/train2/weights/best.pt")
model = YOLO("./runs/detect/train/weights/best.pt")

# 2. Run prediction on an image or video file
# 'show=True' will pop up a window with the boxes drawn
results = model.predict(
    # source="/home/athena/Ultrasound_videos/Ultrasound_Scanning_Supplementary_Video/zoom/Ad_zoom.mp4", show=True, conf=0.5) # Show only
    source="/home/athena/Ultrasound_videos/Ultrasound_Scanning_Supplementary_Video/crop/Smooth_Scan_zoom_v3.mp4", save=True, conf=0.5, device=0) # Save results 
print(f"Video saved to: {results[0].save_dir}")

