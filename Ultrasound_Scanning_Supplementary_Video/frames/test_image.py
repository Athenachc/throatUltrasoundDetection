from ultralytics import YOLO
import os

model = YOLO("./runs/detect/train4/weights/best.pt")

# 2. Path to the image you want to test
image_path = "/home/athena/Ultrasound_videos/Ultrasound_Scanning_Supplementary_Video/frames/dataset_human/train/images/2025_09_30_12_36_55_586.jpg"

# 3. Run prediction
# save=True: saves the result to runs/detect/predictX/
# conf=0.5: only shows detections it is 50% sure about
results = model.predict(source=image_path, save=True, conf=0.5, device=0)

# 4. Print exactly where the image was saved
# results[0].path is the original image path
# results[0].save_dir is the folder where the output is
print(f"Results saved to: {results[0].save_dir}")

# List the files in the output directory to see your image
print("Files in result folder:", os.listdir(results[0].save_dir))