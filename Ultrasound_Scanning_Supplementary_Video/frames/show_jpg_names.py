import os
from pathlib import Path

def list_jpg_in_remote_folder(target_folder_path):
    # Convert to a Path object for easier handling
    folder = Path(target_folder_path).resolve()
    
    # Check if the folder exists
    if not folder.is_dir():
        print(f"Error: The folder '{target_folder_path}' was not found.")
        return

    # Use the folder's name for the .txt file
    folder_name = folder.name
    output_path = folder / f"{folder_name}.txt"

    # Find all .jpg and .jpeg files (case-insensitive)
    jpg_extensions = ('.jpg', '.jpeg', '.JPG', '.JPEG')
    jpg_files = [f.name for f in folder.iterdir() if f.suffix in jpg_extensions]
    
    # Sort them alphabetically
    jpg_files.sort()

    # Write the list to the text file inside that folder
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for filename in jpg_files:
                f.write(filename + '\n')
        print(f"Done! Created '{output_path}' with {len(jpg_files)} filenames.")
    except Exception as e:
        print(f"An error occurred while saving: {e}")

# PASTE YOUR FOLDER PATH HERE
# Example: r'C:\Users\Name\Desktop\Shortlisted' or '/Users/Name/Pictures/Project'
my_path = r'/home/athena/Ultrasound_videos/human_test_results/Process_Data/Data_P_Human_Throat_US/cgl/filtered/20250930122939'  # <-- Change this to your folder path

list_jpg_in_remote_folder(my_path)