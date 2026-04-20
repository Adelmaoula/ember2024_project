import os

# Set HuggingFace cache to Z: drive to avoid filling up C: drive
os.environ["HF_HOME"] = r"Z:\huggingface_cache"

import thrember

def main():
    dataset_dir = r"Z:\ember2024_train_data"
    os.makedirs(dataset_dir, exist_ok=True)
    
    print("Downloading the EMBER2024 train set...")
    # Downloading the full training set
    thrember.download_dataset(dataset_dir, split="train")
    
    print("Vectorizing the downloaded features...")
    from thrember.model import gather_feature_paths, vectorize_subset
    from thrember.features import PEFeatureExtractor
    from pathlib import Path
    
    data_path = Path(dataset_dir)
    X_train_path = data_path / "X_train.dat"
    y_train_path = data_path / "y_train.dat"
    raw_feature_paths = gather_feature_paths(data_path, "train")
    nrows = sum([1 for fp in raw_feature_paths for _ in open(fp)])
    extractor = PEFeatureExtractor()
    vectorize_subset(X_train_path, y_train_path, raw_feature_paths, extractor, nrows)
    
    print("Reading the vectorized features...")
    X_train, y_train = thrember.read_vectorized_features(dataset_dir, subset="train")
    
    print(f"Successfully loaded train set!")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    
    # Print a few labels
    print(f"First 10 labels: {y_train[:10]}")

if __name__ == "__main__":
    main()
