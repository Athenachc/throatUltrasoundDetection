import os
import re
import shutil

def natural_sort_key(s):
    """ Key function for natural sorting (ensures 2.jpg comes before 10.jpg) """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def copy_and_rename_images(source_folder, destination_folder):
    # 1. Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print(f"Created new folder: {destination_folder}")

    # 2. Get the folder name (xxx)
    folder_name = os.path.basename(os.path.normpath(source_folder))
    
    # 3. List and sort images naturally
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    files = [f for f in os.listdir(source_folder) if f.lower().endswith(valid_extensions)]
    files.sort(key=natural_sort_key)

    if not files:
        print("No images found in the source folder.")
        return
    
    # 4. Determine padding (zfill) based on number of files
    # If > 99 files, use 3 digits (001), otherwise 2 digits (01)
    padding = len(str(len(files)))
    if padding < 2: padding = 2

    print(f"Copying {len(files)} images to '{destination_folder}'...")

    # 5. Copy and Rename loop
    for index, filename in enumerate(files, start=1):
        ext = os.path.splitext(filename)[1]
        
        # Create the new name (e.g., xxx_01.jpg)
        new_name = f"ywc_{folder_name}_{str(index).zfill(padding)}{ext}"
        
        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(destination_folder, new_name)
        
        # Copy the file
        shutil.copy2(src_path, dst_path)
        
    print(f"Success! All images copied and renamed in: {destination_folder}")
    print(f"Example filename: ywc_{folder_name}_{'1'.zfill(padding)}{ext}")

# --- SET YOUR PATHS HERE ---
source = "/home/athena/Ultrasound_videos/human_test_results/Process_Data/Data_P_Human_Throat_US/ywc/filtered/20250930121841"
destination = "/home/athena/Ultrasound_videos/Ultrasound_Scanning_Supplementary_Video/frames/dataset_human_v2/train/images"

copy_and_rename_images(source, destination)