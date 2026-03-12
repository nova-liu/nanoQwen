
import os
from huggingface_hub import snapshot_download

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_DIR = "./model"

# --------------------------------
# Step 1 下载模型到 ./model
# --------------------------------
if not os.path.exists(MODEL_DIR):

    print("Downloading model to ./model ...")

    snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False
    )

print("Model path:", MODEL_DIR)