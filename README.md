# throatUltrasoundDetection
## Installation
Prerequisities
- [SAM2](https://github.com/facebookresearch/sam2)
- YOLO
- CV2

## Phantom Data Detection
1. Prepare a dataset with labelling for YOLO. [CVAT](https://www.cvat.ai) is suggested.
2. Train YOLO with [train.py](https://github.com/Athenachc/throatUltrasoundDetection/blob/main/Ultrasound_Scanning_Supplementary_Video/frames/train.py). The YOLO weights will be saved automatically. 
3. Test YOLO results with [test_video.py](https://github.com/Athenachc/throatUltrasoundDetection/tree/main/frames/test_video.py) or [test_image.py](https://github.com/Athenachc/throatUltrasoundDetection/tree/main/frames/test_image.py). Remember to indicate the correct location of the desired YOLO weights.
4. Use SAM2 to draw the boundaries with [yolo_sam2_video_roi_naming_v2.py](https://github.com/Athenachc/throatUltrasoundDetection/tree/main/frames/yolo_sam2_video_roi_naming_v2.py).
   
## Human Data Detection
1. Repeat above Steps 1-3 with corresponding datasets and files locations to prepare YOLO detection.
2. Use SAM2 to draw the boundaries with [yolo_sam2_video_human.py](https://github.com/Athenachc/throatUltrasoundDetection/tree/main/frames/yolo_sam2_video_human.py).
