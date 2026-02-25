import os
import shutil
from pathlib import Path

def merge_cvat_folders(root_path, output_path):
    # Setup final destination
    img_out = Path(output_path) / "images"
    lbl_out = Path(output_path) / "labels"
    
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    global_count = 1
    
    # Get all video folders (xxx, yyy, etc.)
    subfolders = [f.path for f in os.scandir(root_path) if f.is_dir() and f.name != 'combined_dataset']
    subfolders.sort()

    for folder in subfolders:
        folder_path = Path(folder)
        
        # 1. Images are in the 'frames' subfolder
        img_dir = folder_path / "frames"
        
        # 2. Labels are in the 'xxx' folder (same level as 'frames' and 'train.txt')
        lbl_dir = folder_path 
        
        if not img_dir.exists():
            print(f"Skipping {folder}: 'frames' folder not found.")
            continue

        # Filter for images
        images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        for img_name in images:
            # New filename (e.g., 000001.jpg)
            new_name = f"{global_count:06}" 
            ext = os.path.splitext(img_name)[1]
            
            src_img = img_dir / img_name
            dst_img = img_out / f"{new_name}{ext}"
            
            # Match the label name to the image name
            lbl_filename = os.path.splitext(img_name)[0] + ".txt"
            src_lbl = lbl_dir / lbl_filename
            dst_lbl = lbl_out / f"{new_name}.txt"

            if src_img.exists() and src_lbl.exists():
                shutil.copy2(src_img, dst_img)
                shutil.copy2(src_lbl, dst_lbl)
                global_count += 1
            else:
                print(f"Alert: Missing label for {img_name} at {src_lbl}")

    print(f"\nDone! Processed {global_count - 1} total pairs.")
    print(f"Results are in: {os.path.abspath(output_path)}")

# --- CONFIGURATION ---
# Put the path to your 'my_project' folder here
root_directory = "my_project" 
# This is where the 300+ files will end up
destination_directory = "combined_dataset" 

merge_cvat_folders(root_directory, destination_directory)