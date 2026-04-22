import cv2
import os
import glob

def videos_to_images_batch(source_dir, output_root, fps_target=10):
    # Find all common video formats
    video_extensions = ['*.mp4', '*.avi', '*.mkv', '*.MOV']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(source_dir, ext)))

    if not video_files:
        print(f"No videos found in {source_dir}")
        return

    print(f"Found {len(video_files)} videos. Starting extraction at {fps_target} FPS...")

    for video_path in video_files:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_folder = os.path.join(output_root, video_name)
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        if video_fps == 0:
            print(f"Skip: Could not read FPS for {video_name}")
            continue

        # Calculate frame interval for 10 FPS
        # If video is 30fps, interval is 3. If 60fps, interval is 6.
        interval = max(1, round(video_fps / fps_target))
        
        frame_count = 0
        saved_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval == 0:
                # Naming format: videoName_frameNumber.jpg
                file_name = f"{video_name}_{frame_count:05d}.jpg"
                cv2.imwrite(os.path.join(output_folder, file_name), frame)
                saved_count += 1

            frame_count += 1

        cap.release()
        print(f"Completed: {video_name} | Saved {saved_count} images.")

# --- SETTINGS ---
# Path to the folder where your online videos are stored
SOURCE_VIDEOS = "/home/athena/Ultrasound_videos/online_videos/"
# Where you want the image folders to be created
OUTPUT_BASE = "./cvat_upload_10fps"

videos_to_images_batch(SOURCE_VIDEOS, OUTPUT_BASE, fps_target=10)