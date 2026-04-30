import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# --- 1. LOAD DATA ---
DATASET_DIR = r"Z:\ember2024_train_data"
ndim = 2381
nrows_to_read = 60000 

X_path = os.path.join(DATASET_DIR, "X_train.dat")
y_path = os.path.join(DATASET_DIR, "y_train.dat")

print("Reading chunk from disk...")
with open(X_path, 'rb') as f_x:
    X_chunk = np.frombuffer(f_x.read(nrows_to_read * ndim * 4), dtype=np.float32).reshape(nrows_to_read, ndim)
with open(y_path, 'rb') as f_y:
    y_chunk = np.frombuffer(f_y.read(nrows_to_read * 4), dtype=np.int32)

valid_idx = np.where(y_chunk != -1)[0]
X_chunk = X_chunk[valid_idx]
y_chunk = y_chunk[valid_idx]

print("Splitting into Train / Validation...")
X_train, X_valid, y_train, y_valid = train_test_split(X_chunk, y_chunk, test_size=0.2, random_state=42)

del X_chunk
del y_chunk

# --- 2. PREPROCESS ---
print("Cleaning and Scaling Data...")
np.nan_to_num(X_train, copy=False)
np.nan_to_num(X_valid, copy=False)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

# --- 3. APPLY LDA ---
print("Applying LDA to reduce 2,381 features down to 1 optimal dimension...")
lda = LinearDiscriminantAnalysis()
X_train_lda = lda.fit_transform(X_train_scaled, y_train)
X_valid_lda = lda.transform(X_valid_scaled)

# --- 4. TRAIN KNN ---
print("Training KNN on the 1D LDA representation...")
knn = KNeighborsClassifier(n_neighbors=9, n_jobs=-1) # n_jobs=-1 uses all CPU cores
knn.fit(X_train_lda, y_train)

# --- 5. EVALUATE ---
print("Evaluating...")
y_pred = knn.predict(X_valid_lda)
y_prob = knn.predict_proba(X_valid_lda)[:, 1]

print(f"\n===== [ KNN + LDA ] =====")
print(f"Final Accuracy: {accuracy_score(y_valid, y_pred)*100:.2f}%")
print(f"Final AUC:      {roc_auc_score(y_valid, y_prob):.4f}")
