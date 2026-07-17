"""Downloads the IBM Telco Customer Churn dataset from Kaggle and saves it to data/telco_churn.csv.

Usage:
    pip install kagglehub
    python download_data.py

Requires a Kaggle account. On first run, opens a browser for login. Subsequent runs use cached credentials."""

import os
import shutil

try:
    import kagglehub
except ImportError:
    print("Run: pip install kagglehub")
    raise

print("Downloading IBM Telco Customer Churn dataset from Kaggle...")
path = kagglehub.dataset_download("blastchar/telco-customer-churn")

csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
if not csv_files:
    raise FileNotFoundError(f"No CSV found in {path}")

os.makedirs("data", exist_ok=True)
shutil.copy(os.path.join(path, csv_files[0]), "data/telco_churn.csv")

print("Saved to data/telco_churn.csv")
