import os
import torch
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 1. Clear any existing Hydra instances to prevent "Already Initialized" errors
GlobalHydra.instance().clear()

# 2. Manually point Hydra to your current directory (where you copied the .yaml)
# config_path must be relative to this script
initialize(config_path=".", version_base=None)

# 3. Load the config and model
# For config_name, do NOT include the ".yaml" extension
model_cfg_name = "sam2.1_hiera_l" 
sam2_checkpoint = "/home/athena/sam2/checkpoints/sam2.1_hiera_large.pt"

print("Building SAM2.1 model...")
sam2_model = build_sam2(model_cfg_name, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)

print("Success! SAM2.1 is ready on your RTX 5070 Ti.")