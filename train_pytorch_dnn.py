import os
import sys
import numpy as np
import time

# --- ENVIRONMENT SETUP ---
base_dir = os.getcwd()
for venv_folder in ["venv", ".venv"]:
    site_packages = os.path.join(base_dir, venv_folder, "Lib", "site-packages")
    if os.path.exists(site_packages) and site_packages not in sys.path:
        sys.path.insert(0, site_packages)
        break

global_venv_site_packages = r"Z:\ai project\.venv\Lib\site-packages"
if os.path.exists(global_venv_site_packages) and global_venv_site_packages not in sys.path:
    sys.path.insert(0, global_venv_site_packages)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    print(f"PyTorch version {torch.__version__} loaded.")
except ImportError:
    print("PyTorch is not installed. Please run: pip install torch torchvision torchaudio")
    sys.exit(1)

try:
    from thrember.features import PEFeatureExtractor
    ndim = PEFeatureExtractor.dim
except ImportError:
    ndim = 2381

# --- CONFIGURATION ---
DATASET_DIR = r"Z:\ember2024_train_data"
CHUNK_SIZE = 250000              # Large RAM chunk to safely pull from disk
BATCH_SIZE = 4096                # Mini-batch size for GPU/CPU optimization
EPOCHS = 10                      # Number of full passes over dataset
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using compute device: {DEVICE}")

# --- MODEL ARCHITECTURE ---
class MalwareDNN(nn.Module):
    def __init__(self, input_dim):
        super(MalwareDNN, self).__init__()
        
        # Deep Feedforward Neural Network optimized for high-dimensional tabular data
        self.net = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 1) 
            # Note: No Sigmoid here! We use BCEWithLogitsLoss which applies it internally for better numeric stability.
        )

    def forward(self, x):
        return self.net(x)

# --- TRAINING PIPELINE ---
def train_dnn():
    X_path = os.path.join(DATASET_DIR, "X_train.dat")
    y_path = os.path.join(DATASET_DIR, "y_train.dat")
    
    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError("Training data files not found.")

    file_size = os.path.getsize(X_path)
    nrows = file_size // (ndim * 4)
    train_nrows = int(nrows * 0.9)  # Reserve last 10% for test
    
    X_memmap = np.memmap(X_path, dtype=np.float32, mode="r", shape=(nrows, ndim))
    y_memmap = np.memmap(y_path, dtype=np.int32, mode="r", shape=(nrows,))

    model = MalwareDNN(input_dim=ndim).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    print("\nStarting Out-of-Core Batch Training...")
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        processed_samples = 0
        start_time = time.time()
        
        # Stream data in massive chunks to avoid RAM crashes
        for start_idx in range(0, train_nrows, CHUNK_SIZE):
            end_idx = min(start_idx + CHUNK_SIZE, train_nrows)
            
            # Load exact chunk into RAM
            X_chunk = np.array(X_memmap[start_idx:end_idx])
            y_chunk = np.array(y_memmap[start_idx:end_idx])
            
            # Filter unlabeled
            valid_idx = y_chunk != -1
            X_chunk, y_chunk = X_chunk[valid_idx], y_chunk[valid_idx]
            
            if len(y_chunk) == 0:
                continue
                
            # Convert to PyTorch tensors
            X_tensor = torch.tensor(X_chunk, dtype=torch.float32)
            y_tensor = torch.tensor(y_chunk, dtype=torch.float32).unsqueeze(1) # [batch_size, 1]
            
            # Create DataLoader for this specific RAM chunk
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            
            # Mini-batch training
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * batch_X.size(0)
                processed_samples += batch_X.size(0)
                
            print(f"  [Epoch {epoch+1}] Indexed up to row {end_idx:,} / {train_nrows:,}...")

        avg_loss = epoch_loss / processed_samples
        elapsed = time.time() - start_time
        print(f"=> EPOCH {epoch+1} COMPLETE | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.2f} sec")
        
        # Step learning rate scheduler
        scheduler.step(avg_loss)
        
        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(DATASET_DIR, "ember_dnn_checkpoint.pth"))
        
    print("\nTraining Complete! Final model saved to 'ember_dnn_checkpoint.pth'.")

if __name__ == "__main__":
    train_dnn()
