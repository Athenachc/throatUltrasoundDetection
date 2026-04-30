from pathlib import Path

def list_ground_truth_filenames(target_folder_path):
    folder = Path(target_folder_path).resolve()
    
    if not folder.is_dir():
        print(f"Error: Folder '{target_folder_path}' not found.")
        return

    # Updated naming convention based on your previous preference
    output_filename = f"{folder.name}_shortlisted.txt"
    output_path = folder / output_filename

    # Capture all common image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG')
    img_files = [f.name for f in folder.iterdir() if f.suffix in valid_extensions]
    img_files.sort()

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for filename in img_files:
                f.write(filename + '\n')
        print(f"Success! {len(img_files)} files listed in: {output_path}")
    except Exception as e:
        print(f"Save failed: {e}")

# Target GT folder
my_path = r'/home/athena/Ultrasound_videos/Ultrasound_Scanning_Supplementary_Video/frames/online_human_selected_GT_label/SegmentationClass/'
list_ground_truth_filenames(my_path)