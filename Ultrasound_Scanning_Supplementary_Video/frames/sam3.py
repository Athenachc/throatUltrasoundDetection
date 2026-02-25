from modelscope import snapshot_download

# This will download the weights to a local folder
model_dir = snapshot_download('facebook/sam3')
print(f"Model downloaded to: {model_dir}")