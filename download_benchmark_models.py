import os
from huggingface_hub import snapshot_download

def download_benchmark_models():
    # Target directory
    destination_dir = r"Z:\ember2024_train_data\benchmark_models"
    os.makedirs(destination_dir, exist_ok=True)
    
    print(f"Downloading benchmark models to {destination_dir}...")
    
    try:
        # Download the entire repository
        path = snapshot_download(
            repo_id="joyce8/EMBER2024-benchmark-models",
            repo_type="model",
            local_dir=destination_dir,
            local_dir_use_symlinks=False
        )
        print(f"Successfully downloaded models to: {path}")
        
        # List files
        print("Files downloaded:")
        for root, dirs, files in os.walk(path):
            for file in files:
                print(f" - {file}")
                
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    download_benchmark_models()
