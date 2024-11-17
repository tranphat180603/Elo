from datasets import load_dataset
from transformers import snapshot_download
import os

print(f"Number of files before downloading existing dataset {len(os.listdir("processed_images"))}")


download_patterns = ["*.png", "*.jpeg", "*.jpg"]
#Load the subdirectories of Omni Parser into weights
snapshot_download(
    repo_id="tranphat1806/UI-tron",
    repo_type = "dataset",
    local_dir ="/processed_images",
    allow_patterns = download_patterns,
)
print(f"Number of files after downloading existing dataset {len(os.listdir("processed_images"))}")

#push entire folder to hub
ds = load_dataset("imagefolder", data_dir = "/processed_images")
ds.push_to_hub("tranphat1806/UI-tron")