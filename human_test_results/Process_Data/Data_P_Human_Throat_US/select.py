import os
import shutil
import re

# --- CONFIGURATION ---
SOURCE_FOLDER = "/home/athena/Ultrasound_videos/human_test_results/Process_Data/Data_P_Human_Throat_US/ywc/filtered/20250930121841"
DEST_FOLDER = "/home/athena/Ultrasound_videos/human_test_results/Process_Data/Data_P_Human_Throat_US/selected_images"
INTERVAL = 14 

def natural_sort_key(s):
    """Sorts strings containing numbers in the way humans expect (1, 2, 10, 11...)"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def subsample_images():
    # 1. Create destination if it doesn't exist
    if not os.path.exists(DEST_FOLDER):
        os.makedirs(DEST_FOLDER)
        print(f"Created folder: {DEST_FOLDER}")

    # 2. Get and sort image list
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    all_files = [f for f in os.listdir(SOURCE_FOLDER) if f.lower().endswith(extensions)]
    all_files.sort(key=natural_sort_key)

    if not all_files:
        print("No images found in source folder.")
        return

    print(f"Found {len(all_files)} images. Copying every {INTERVAL}th image...")

    # 3. Loop and Copy
    count = 0
    # Start at index INTERVAL-1 (e.g., index 9 is the 10th image)
    for i in range(INTERVAL - 1, len(all_files), INTERVAL):
        filename = all_files[i]
        src_path = os.path.join(SOURCE_FOLDER, filename)
        dst_path = os.path.join(DEST_FOLDER, filename)
        
        shutil.copy2(src_path, dst_path) # copy2 preserves metadata
        count += 1
        print(f"Copied: {filename}")

    print(f"\nDone! Copied {count} images to {DEST_FOLDER}")

if __name__ == "__main__":
    subsample_images()