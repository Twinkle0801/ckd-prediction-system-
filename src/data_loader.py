# src/data_loader.py
import kagglehub
import shutil
import os

def fetch_ckd_dataset(output_path: str = "data/raw/kidney_disease.csv"):
    path = kagglehub.dataset_download("mansoordaku/ckdisease")
    print("Downloaded to:", path)

    for f in os.listdir(path):
        if f.endswith(".csv"):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy(os.path.join(path, f), output_path)
            print(f"Saved to {output_path}")
            return output_path

if __name__ == "__main__":
    fetch_ckd_dataset()