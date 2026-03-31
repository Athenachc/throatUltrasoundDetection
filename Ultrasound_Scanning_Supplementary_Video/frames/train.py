from ultralytics import YOLO
import os

def main():
    # 1. Path to your data.yaml
    # Make sure this path is correct relative to where you run the script!
    yaml_path = "./dataset_human_v2/data.yaml" 
    
    if not os.path.exists(yaml_path):
        print(f"ERROR: Cannot find {yaml_path}. Check your folder structure!")
        return

    # 2. Load the model
    print("Loading YOLOv8 Nano model...")
    model = YOLO("yolov8n.pt") # Current baseline
    # model = YOLO("yolov8s.pt")   # Small (more accurate than nano)
    # model = YOLO("yolo11n.pt")   # Latest version
    # model = YOLO("yolov10n.pt")  # NMS-free stability

    # 3. Start Training
    print("Starting training...")
    model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,      # Use 0 for your NVIDIA GPU
        plots=True     # This saves accuracy charts in the 'runs' folder
    )
    print("Training complete!")

if __name__ == "__main__":
    main()