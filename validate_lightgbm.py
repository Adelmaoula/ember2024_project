import os
import sys
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
from thrember.features import PEFeatureExtractor

# Add local venv if needed (copying safety from fraction_training)
venv_site_packages = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "Lib", "site-packages")
if os.path.exists(venv_site_packages):
    sys.path.append(venv_site_packages)

def evaluate_model(dataset_dir, model_filename="lightgbm_model.txt", batch_size=100000):
    print(f"Preparing to evaluate model from {dataset_dir}...")
    
    model_path = os.path.join(dataset_dir, model_filename)
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # Load Model
    print(f"Loading model: {model_filename}...")
    model = lgb.Booster(model_file=model_path)

    X_path = os.path.join(dataset_dir, "X_train.dat")
    y_path = os.path.join(dataset_dir, "y_train.dat")
    plots_dir = os.path.join(dataset_dir, "evaluation_plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Get dimensions
    extractor = PEFeatureExtractor()
    ndim = extractor.dim
    file_size = os.path.getsize(X_path)
    nrows = file_size // (ndim * 4)
    
    # Calculate Split
    train_nrows = int(nrows * 0.9)
    val_nrows = nrows - train_nrows
    val_start_idx = train_nrows
    
    print(f"Total rows: {nrows}")
    print(f"Validation set size: {val_nrows} samples (starts at index {val_start_idx})")

    # Mmap data
    X_memmap = np.memmap(X_path, dtype=np.float32, mode="r", shape=(nrows, ndim))
    y_memmap = np.memmap(y_path, dtype=np.int32, mode="r", shape=(nrows,))

    # BATched Prediction to avoid RAM explosion
    print("Running predictions in batches...")
    
    y_true_all = []
    y_pred_prob_all = []

    for batch_start in range(val_start_idx, nrows, batch_size):
        batch_end = min(batch_start + batch_size, nrows)
        
        # Load batch
        X_batch = np.array(X_memmap[batch_start:batch_end])
        y_batch = np.array(y_memmap[batch_start:batch_end])
        
        # Filter unlabeled
        valid_mask = y_batch != -1
        if not np.any(valid_mask):
            continue
            
        X_batch = X_batch[valid_mask]
        y_batch = y_batch[valid_mask]
        
        # Predict
        preds = model.predict(X_batch)
        
        y_true_all.append(y_batch)
        y_pred_prob_all.append(preds)
        
        print(f"Processed batch {batch_start} - {batch_end}")

    # Concatenate results
    y_val = np.concatenate(y_true_all)
    y_pred_prob = np.concatenate(y_pred_prob_all)
    y_pred = (y_pred_prob > 0.5).astype(int)

    print(f"\nExample Predictions vs Actual:")
    print(f"Pred: {y_pred[:10]}")
    print(f"True: {y_val[:10]}")

    # Metrics
    auc = roc_auc_score(y_val, y_pred_prob)
    acc = accuracy_score(y_val, y_pred)
    
    print("-" * 30)
    print(f"Validation AUC:      {auc:.4f}")
    print(f"Validation Accuracy: {acc:.4f}")
    print("-" * 30)

    # PLOTS
    print(f"Generating plots in {plots_dir}...")
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'LightGBM (AUC = {auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'roc_curve.png'))
    plt.close()

    # 2. Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malicious'], 
                yticklabels=['Benign', 'Malicious'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
    plt.close()

    # 3. Feature Importance
    plt.figure(figsize=(12, 10))
    lgb.plot_importance(model, max_num_features=30, title='Top 30 Feature Importances', 
                        importance_type='split', figsize=(12, 10))
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_importance.png'))
    plt.close()

    print("Evaluation complete.")

if __name__ == "__main__":
    dataset_dir = r"Z:\ember2024_train_data"
    evaluate_model(dataset_dir, model_filename="ember_model_full.txt")
