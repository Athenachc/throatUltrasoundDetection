import cv2
import os
import re

# --- CONFIGURATION ---
# Just change this path; the output name will follow the folder name
FOLDER_PATH = "/home/athena/Ultrasound_videos/human_test_results/0930-cuhk-002/cgl/20250930123155" 
# FOLDER_PATH = "/home/athena/Ultrasound_videos/human_test_results/Process_Data/Data_P_Human_Throat_US/cgl/filtered/20250930123155" 20250930122939
FPS = 10.0  

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def convert_folder_to_video():
    # 1. Get the folder name to create the video name
    # e.g., if path is /home/user/xxx, folder_name becomes 'xxx'
    folder_name = os.path.basename(os.path.normpath(FOLDER_PATH))
    # output_video_name = f"{folder_name}_shortlisted_v2.mp4"
    output_video_name = f"{folder_name}.mp4"

    # 2. Get and sort images
    images = [img for img in os.listdir(FOLDER_PATH) if img.lower().endswith((".jpg", ".jpeg", ".png"))]
    images.sort(key=natural_sort_key)

    if not images:
        print(f"No images found in {FOLDER_PATH}")
        return

    # 3. Get dimensions from first image
    first_frame = cv2.imread(os.path.join(FOLDER_PATH, images[0]))
    h, w, _ = first_frame.shape
    size = (w, h)

    # 4. Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_name, fourcc, FPS, size)

    print(f"Creating video: {output_video_name}")
    print(f"Source Folder: {folder_name}")
    
    for i, filename in enumerate(images):
        img_path = os.path.join(FOLDER_PATH, filename)
        frame = cv2.imread(img_path)
        
        if frame is None:
            print(f"Skipping corrupted frame: {filename}")
            continue
            
        out.write(frame)
        
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(images)} frames written...", end="\r")

    out.release()
    print(f"\nSuccess! Video saved as: {os.getcwd()}/{output_video_name}")

if __name__ == "__main__":
    convert_folder_to_video()