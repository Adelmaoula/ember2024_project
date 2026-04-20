import os
import sys
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, classification_report  # Updated import

# Add local venv site-packages to path
venv_site_packages = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv", "Lib", "site-packages")
if os.path.exists(venv_site_packages):
    sys.path.append(venv_site_packages)
    
from thrember.features import PEFeatureExtractor

def evaluate_benchmark(dataset_dir, model_path, batch_size=100000):
    print(f"Preparing to evaluate benchmark model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # Load Model
    print(f"Loading LightGBM model from {model_path}...")
    try:
        model = lgb.Booster(model_file=model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    X_path = os.path.join(dataset_dir, "X_train.dat")
    y_path = os.path.join(dataset_dir, "y_train.dat")
    
    # Store plots in a specific benchmark folder
    model_name = os.path.basename(model_path).replace(".model", "").replace(".txt", "")
    plots_dir = os.path.join(dataset_dir, f"benchmark_plots_{model_name}")
    os.makedirs(plots_dir, exist_ok=True)

    # Get dimensions
    try:
        extractor = PEFeatureExtractor()
        ndim = extractor.dim
    except:
        ndim = 2381 # Default fallback if extractor fails
        print(f"Warning: Could not load feature extractor. Using default dim: {ndim}")

    if not os.path.exists(X_path):
        print(f"Error: Data file not found at {X_path}")
        return
        
    file_size = os.path.getsize(X_path)
    nrows = file_size // (ndim * 4)
    
    # Use the validation split (last 10%)
    train_nrows = int(nrows * 0.9)
    val_nrows = nrows - train_nrows
    val_start_idx = train_nrows
    
    print(f"Total rows in dataset: {nrows}")
    print(f"Evaluating on the validation set (last 10%): {val_nrows} samples starting at index {val_start_idx}")

    # Mmap data
    X_memmap = np.memmap(X_path, dtype=np.float32, mode="r", shape=(nrows, ndim))
    y_memmap = np.memmap(y_path, dtype=np.int32, mode="r", shape=(nrows,))

    print("Running predictions in batches...")
    
    y_true_all = []
    y_pred_prob_all = []

    # Process in batches to save RAM
    for batch_start in range(val_start_idx, nrows, batch_size):
        batch_end = min(batch_start + batch_size, nrows)
        
        # Load batch
        X_batch = np.array(X_memmap[batch_start:batch_end])
        y_batch = np.array(y_memmap[batch_start:batch_end])
        
        # Filter unlabeled
        valid_mask = y_batch != -1
        
        # Also filter out rows that might be corrupted or empty if any
        if not np.any(valid_mask):
            continue
            
        X_batch = X_batch[valid_mask]
        y_batch = y_batch[valid_mask]
        
        if len(y_batch) == 0:
            continue

        # Predict
        try:
            preds = model.predict(X_batch)
            y_true_all.append(y_batch)
            y_pred_prob_all.append(preds)
        except Exception as e:
            print(f"Prediction error on batch {batch_start}: {e}")
            continue
        
        print(f"Processed batch {batch_start} - {batch_end}")

    if not y_true_all:
        print("No valid data found for evaluation.")
        return

    # Concatenate results
    y_val = np.concatenate(y_true_all)
    y_pred_prob = np.concatenate(y_pred_prob_all)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Metrics
    auc = roc_auc_score(y_val, y_pred_prob)
    acc = accuracy_score(y_val, y_pred)
    
    print("-" * 40)
    print(f"RESULTS FOR MODEL: {model_name}")
    print(f"Validation AUC:      {auc:.4f}")
    print(f"Validation Accuracy: {acc:.4f}")
    print("-" * 40)
    
    # Save Classification Report
    try:
        report = classification_report(y_val, y_pred, target_names=['Benign', 'Malicious'])
        print(report)
        with open(os.path.join(plots_dir, "classification_report.txt"), "w") as f:
            f.write(report)
    except Exception as e:
        print(f"Could not generate classification report: {e}")

    # PLOTS
    print(f"Generating plots in {plots_dir}...")
    
    # 1. ROC Curve
    try:
        fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, label=f'Model AUC = {auc:.4f}', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
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
        plt.title(f'Confusion Matrix - {model_name}')
        plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
        plt.close()
    except Exception as e:
        print(f"Error generating plots: {e}")

    print(f"Evaluation of {model_name} complete.")

if __name__ == "__main__":
    dataset_dir = r"Z:\ember2024_train_data"
    
    # Evaluate the ALL model (General Purpose)
    # Ensure this path matches where you put the models
    benchmark_model_path = r"C:\Users\him\ember2024_project\benchmark_models\EMBER2024_all.model"
    
    evaluate_benchmark(dataset_dir, benchmark_model_path)

    # Mmap data
    X_memmap = np.memmap(X_path, dtype=np.float32, mode="r", shape=(nrows, ndim))
    y_memmap = np.memmap(y_path, dtype=np.int32, mode="r", shape=(nrows,))

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

    # Metrics
    auc = roc_auc_score(y_val, y_pred_prob)
    acc = accuracy_score(y_val, y_pred)
    
    print("-" * 30)
    print(f"BENCHMARK MODEL RESULTS")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Validation AUC:      {auc:.4f}")
    print(f"Validation Accuracy: {acc:.4f}")
    print("-" * 30)

    # PLOTS
    print(f"Generating plots in {plots_dir}...")
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_pred_prob)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'Benchmark (AUC = {auc:.4f})', linewidth=2, color='darkorange')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {os.path.basename(model_path)}')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'benchmark_roc_curve.png'))
    plt.close()

    # 2. Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
                xticklabels=['Benign', 'Malicious'], 
                yticklabels=['Benign', 'Malicious'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {os.path.basename(model_path)}')
    plt.savefig(os.path.join(plots_dir, 'benchmark_confusion_matrix.png'))
    plt.close()

    print("Evaluation complete.")

if __name__ == "__main__":
    dataset_dir = r"Z:\ember2024_train_data"
    
    # Evaluate the ALL model (General Purpose)
    # Ensure this path matches where you put the models
    benchmark_model_path = r"C:\Users\him\ember2024_project\benchmark_models\EMBER2024_all.model"
    
    evaluate_benchmark(dataset_dir, benchmark_model_path)
