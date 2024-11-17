from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download, login
import os
import shutil

# num_files = len(os.listdir("processed_images"))
# print(f"Number of files before downloading existing dataset {num_files}")
    
# download_patterns = ["*.png", "*.jpeg", "*.jpg"]
# #Load the subdirectories of Omni Parser into weights
# snapshot_download(
#     repo_id="tranphat1806/UI-tron",
#     repo_type = "dataset",
#     local_dir ="/processed_images",
#     allow_patterns = download_patterns,
# )

# num_files = len(os.listdir("processed_images"))
# print(f"Number of files after downloading existing dataset {num_files}")

#push entire folder to hub
ds = load_dataset("imagefolder", data_dir = "processed_images")
ds.push_to_hub("tranphat1806/UI-tron")