import os
import sys
import numpy as np
import lightgbm as lgb

# Add local venv site-packages to path so thrember can be imported
base_dir = os.getcwd()
if os.path.exists(os.path.join(base_dir, "venv", "Lib", "site-packages")):
    sys.path.append(os.path.join(base_dir, "venv", "Lib", "site-packages"))
elif os.path.exists(os.path.join(base_dir, ".venv", "Lib", "site-packages")):
    sys.path.append(os.path.join(base_dir, ".venv", "Lib", "site-packages"))

try:
    from thrember.features import PEFeatureExtractor
    print("Features extractor imported successfully.")
except ImportError:
    print("Warning: 'thrember' not found. Will use default feature dimension (2381).")
    class PEFeatureExtractor:
        dim = 2381

def train_model(dataset_dir, chunk_size=10000):
    print(f"Preparing to train from {dataset_dir}...")

    X_path = os.path.join(dataset_dir, "X_train.dat")
    y_path = os.path.join(dataset_dir, "y_train.dat")

    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError("Error: Training data not found!")

    # Get the feature dimension
    extractor = PEFeatureExtractor()
    ndim = extractor.dim

    # Calculate sizing
    file_size = os.path.getsize(X_path)
    nrows = file_size // (ndim * 4)

    print(f"Total samples detected: {nrows}")
    print(f"Feature dimensions: {ndim}")

    # Open files in read-only memmap mode
    X_memmap = np.memmap(X_path, dtype=np.float32, mode="r", shape=(nrows, ndim))
    y_memmap = np.memmap(y_path, dtype=np.int32, mode="r", shape=(nrows,))

    # Split: 90% Training, 10% Validation
    train_nrows = int(nrows * 0.9)
    val_nrows = nrows - train_nrows
    print(f"Training on {train_nrows} samples")
    print(f"Validating on {val_nrows} samples (Skipped in this training loop)")

    # LightGBM parameters optimized for EMBER
    params = {
        "objective": "binary",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 1024,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "verbose": -1,
        "n_jobs": -1  # Use all CPU cores
    }

    # Indices of categorical features as defined by the EMBER dataset
    categorical_features = [2, 3, 4, 5, 6, 701, 702]

    model = None
    start_chunk_idx = 0

    # RETRIEVE STATE LOGIC
    ckpt_path = os.path.join(dataset_dir, "ember_model_checkpoint.txt")
    state_path = os.path.join(dataset_dir, "ember_training_state.txt")

    if os.path.exists(ckpt_path) and os.path.exists(state_path):
        print(f"Found checkpoint! Resuming training from {ckpt_path}...")
        try:
            model = lgb.Booster(model_file=ckpt_path)
            with open(state_path, "r") as f:
                start_chunk_idx = int(f.read().strip())
            print(f"Resuming from chunk index: {start_chunk_idx}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
            model = None
            start_chunk_idx = 0
    else:
        print("No checkpoint found. Starting from scratch.")

    print("\n" + "="*40)
    print("🚀 TRAINING STARTED...")
    print("="*40)
    print(f"Starting training in chunks of {chunk_size}...")

    # Calculate total chunks for progress tracking
    total_chunks = (train_nrows + chunk_size - 1) // chunk_size
    current_chunk = 0

    for start_idx in range(0, train_nrows, chunk_size):
        # RESUME LOGIC: Skip chunks we've already processed
        if current_chunk < start_chunk_idx:
            current_chunk += 1
            print(f"Skipping chunk {current_chunk-1} (already trained)")
            continue

        end_idx = min(start_idx + chunk_size, train_nrows)
        print(f"Training chunk {current_chunk}/{total_chunks}: {start_idx} to {end_idx}...")
        
        # Load ONLY this chunk into RAM
        X_chunk = np.array(X_memmap[start_idx:end_idx])
        y_chunk = np.array(y_memmap[start_idx:end_idx])
        
        # Filter out unlabeled data (-1)
        valid_idx = y_chunk != -1
        X_chunk = X_chunk[valid_idx]
        y_chunk = y_chunk[valid_idx]
        
        if len(y_chunk) == 0:
            print("  Skipping empty chunk")
            current_chunk += 1
            continue
            
        # Create LightGBM dataset for this chunk
        train_data = lgb.Dataset(
            X_chunk, 
            label=y_chunk, 
            categorical_feature=categorical_features,
            free_raw_data=False
        )
        
        # Train incrementally
        model = lgb.train(
            params,
            train_data,
            num_boost_round=50, 
            init_model=model,  
            keep_training_booster=True
        )
        
        current_chunk += 1
        
        # SAVE STATE LOGIC: Save checkpoint every 5 chunks
        if current_chunk % 5 == 0:
            model.save_model(ckpt_path)
            with open(state_path, "w") as f:
                f.write(str(current_chunk))
            print(f"  [Checkpoint saved to {ckpt_path}. Next start index: {current_chunk}]")

    print("\nTraining Loop Complete!")

    # Save the final model
    save_path = os.path.join(dataset_dir, "ember_model_full.txt")
    print(f"Saving model to {save_path}...")
    model.save_model(save_path)
    print("Model saved successfully!")

if __name__ == "__main__":
    # Point this to your Z: drive
    DATASET_DIR = r"Z:\ember2024_train_data"
    CHUNK_SIZE = 10000
    
    train_model(DATASET_DIR, CHUNK_SIZE)