import os
import sys
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve

# --- Environment Setup ---
# Add local venv site-packages to path so thrember can be imported if running outside activated venv
venv_site_packages = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "Lib", "site-packages")
if os.path.exists(venv_site_packages):
    sys.path.append(venv_site_packages)

try:
    from thrember.features import PEFeatureExtractor
    print("Features extractor imported successfully.")
except ImportError:
    print("Warning: 'thrember' not found. Will use default feature dimension (2381).")
    class PEFeatureExtractor:
        dim = 2381

def evaluate_benchmark_model(model_path, dataset_dir, batch_size=50000):
    model_name = os.path.basename(model_path)
    print(f"\n{'='*50}")
    print(f"EVALUATING BENCHMARK MODEL: {model_name}")
    print(f"{'='*50}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # 1. Load Model
    print(f"Loading LightGBM model from: {model_path}...")
    try:
        model = lgb.Booster(model_file=model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 2. Setup Data Paths
    X_path = os.path.join(dataset_dir, "X_train.dat")
    y_path = os.path.join(dataset_dir, "y_train.dat")
    
    # Create output directory for stats/plots
    output_dir = os.path.join(dataset_dir, "benchmark_stats", model_name.replace('.model', ''))
    os.makedirs(output_dir, exist_ok=True)
    print(f"Validation results will be saved to: {output_dir}")

    # 3. Validation Split Logic
    extractor = PEFeatureExtractor()
    ndim = extractor.dim
    
    if not os.path.exists(X_path):
         print(f"Data file not found: {X_path}")
         return
         
    file_size = os.path.getsize(X_path)
    nrows = file_size // (ndim * 4) # float32 = 4 bytes
    
    # Matches training split: Train 90%, Val 10%
    train_nrows = int(nrows * 0.9)
    val_nrows = nrows - train_nrows
    val_start_idx = train_nrows
    
    print(f"Total dataset rows: {nrows}")
    print(f"Validation set size: {val_nrows} samples (Index {val_start_idx} to {nrows})")

    # 4. Memory Mapping
    # Using 'r' mode to ensure we don't accidentally modify data
    X_memmap = np.memmap(X_path, dtype=np.float32, mode="r", shape=(nrows, ndim))
    y_memmap = np.memmap(y_path, dtype=np.int32, mode="r", shape=(nrows,))

    # 5. Batch Prediction Loop
    print(f"Starting batched prediction (Batch Size: {batch_size})...")
    
    all_y_true = []
    all_y_pred = []
    
    num_batches = (val_nrows + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start = val_start_idx + i * batch_size
        end = min(start + batch_size, nrows)
        current_len = end - start
        
        # Copy batch to RAM for speed during prediction
        # (LightGBM might copy anyway, but explicit copy ensures continuous memory)
        X_batch = np.array(X_memmap[start:end])
        y_batch = np.array(y_memmap[start:end])
        
        # Filter out ignored labels (-1) immediately to save processing
        # In Ember 2018 validation set there are -1s. In 2024 potentially too.
        valid_mask = y_batch != -1
        
        if np.any(valid_mask):
            X_batch_valid = X_batch[valid_mask]
            y_batch_valid = y_batch[valid_mask]
            
            # Predict
            preds = model.predict(X_batch_valid)
            
            all_y_true.extend(y_batch_valid)
            all_y_pred.extend(preds)
        
        if (i+1) % 5 == 0 or (i+1) == num_batches:
             print(f"  Processed batch {i+1}/{num_batches}...")

    # 6. Calculate Metrics
    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    
    if len(y_true) == 0:
        print("Error: No valid labels found in validation set!")
        return

    print(f"Calculating metrics on {len(y_true)} samples...")
    
    auc = roc_auc_score(y_true, y_pred)
    # Using 0.5 threshold for accuracy/confusion matrix
    acc = accuracy_score(y_true, y_pred > 0.5)
    
    print(f"Validation AUC:      {auc:.5f}")
    print(f"Validation Accuracy: {acc:.5f} (threshold 0.5)")

    # 7. Save Results
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        f.write(f"Model File: {model_name}\n")
        f.write(f"Validation Samples: {len(y_true)}\n")
        f.write(f"AUC: {auc:.6f}\n")
        f.write(f"Accuracy: {acc:.6f}\n")
        
    print("Generatng plots...")
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {model_name}')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()

    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred[y_true==0], bins=50, alpha=0.5, label='Benign', density=True)
    plt.hist(y_pred[y_true==1], bins=50, alpha=0.5, label='Malicious', density=True)
    plt.title(f'Score Distribution: {model_name}')
    plt.xlabel('Malicious Score')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "dist.png"))
    plt.close()

    print(f"Done. Stats saved to {output_dir}")

if __name__ == "__main__":
    # Define paths
    DATASET_DIR = r"Z:\ember2024_train_data"
    MODELS_DIR = r"C:\Users\him\ember2024_project\benchmark_models"
    
    if not os.path.exists(MODELS_DIR):
        print(f"Directory not found: {MODELS_DIR}")
        sys.exit(1)
        
    # Find all .model files
    models = [f for f in os.listdir(MODELS_DIR) if f.endswith(".model")]
    
    if not models:
        print("No .model files found in benchmark directory.")
    else:
        print(f"Found {len(models)} benchmark models: {models}")
        
        # Evaluate each one
        for m in models:
            m_path = os.path.join(MODELS_DIR, m)
            evaluate_benchmark_model(m_path, DATASET_DIR)
