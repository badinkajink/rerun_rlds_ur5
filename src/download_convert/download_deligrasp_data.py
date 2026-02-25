import os
from huggingface_hub import snapshot_download

def download_specific_folders():
    repo_id = "correlllab/justaddforce-data"
    local_download_dir = "./justaddforce_dataset"
    

    target_folders = [
        "deligrasp_dataset/*",
        "deligrasp_dataset_grasponly/*"
    ]
    
    print(f"Initiating targeted download from {repo_id}...")
    
    try:
        download_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=target_folders,       
            local_dir=local_download_dir, 
            local_dir_use_symlinks=False
        )
        print(f"\nSuccess! Targeted folders downloaded to: {os.path.abspath(download_path)}")
        
    except Exception as e:
        print(f"\nAn error occurred during the download: {e}")

if __name__ == "__main__":
    download_specific_folders()