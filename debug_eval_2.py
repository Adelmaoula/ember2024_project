import os
import sys

print("Initializing...", flush=True)

# Add local venv site-packages to path so thrember can be imported
venv_site_packages = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Lib", "site-packages")
print(f"Adding to path: {venv_site_packages}", flush=True)

if os.path.exists(venv_site_packages):
    sys.path.append(venv_site_packages)
else:
    print(f"WARNING: {venv_site_packages} not found!", flush=True)

try:
    import numpy as np
    print("NumPy imported.", flush=True)
    import lightgbm as lgb
    print("LightGBM imported.", flush=True)
    
    # We might need thrember later, but let's check basic structure first
    try:
        from thrember.features import PEFeatureExtractor
        print("EMBER Feature Extractor imported.", flush=True)
    except ImportError:
        print("WARNING: thrember module not found, continuing without it.", flush=True)

except Exception as e:
    print(f"Import error: {e}", flush=True)
    sys.exit(1)

# Using user's path structure based on previous interactions
model_path = r"Z:\ember2024_train_data\benchmark_models\EMBER2024_all.model"
# If Z: drive is the issue, fallback to relative check (though user context says Z:)
if not os.path.exists(model_path):
    print(f"Model file missing at: {model_path}", flush=True)
    # Check current directory just in case
    local_path = "EMBER2024_all.model"
    if os.path.exists(local_path):
        model_path = local_path
        print(f"Found local model: {model_path}")
else:
    print(f"Model file found: {model_path}", flush=True)
    try:
        model = lgb.Booster(model_file=model_path)
        print("Model loaded successfully.", flush=True)
    except Exception as e:
        print(f"Model load error: {e}", flush=True)
