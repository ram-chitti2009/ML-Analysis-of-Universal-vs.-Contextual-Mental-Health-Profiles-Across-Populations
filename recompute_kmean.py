"""
Recompute KMeans centroids from a trained AE checkpoint (.pth) and data.
Run in Colab/local CPU. Update the paths, FEATURES, and k before running.
"""

import os
import pickle
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


DATA_CONFIG = {
    "D1-Swiss": {
        "model_path": "D1_Swiss_model (3).pth",
        "data_path": "D1_Swiss_processed.csv",
        "k": 2,
        "features": ["Depression", "Anxiety", "Stress", "Burnout"],
    },
    "D2-Cultural": {
        "model_path": "D2_Cultural_model (2).pth",  # adjust if filename differs
        "data_path": "D2_Cultural_processed.csv",
        "k": 6,
        "features": ["Depression", "Anxiety", "Stress", "Burnout"],
    },
    "D3-Academic": {
        "model_path": "D3_Academic_model (2).pth",  # note the space in filename
        "data_path": "D3_Academic_processed.csv",
        "k": 2,
        "features": ["Depression", "Anxiety", "Stress", "Burnout"],
    },
    "D4-Tech": {
        "model_path": "D4_Tech_model (2).pth",
        "data_path": "D4_Tech_processed.csv",
        "k": 3,
        "features": ["Depression", "Anxiety", "Stress", "Burnout"],
    },
}

def run_one(name, cfg, save_dir="/content"):
    print(f"\n=== Processing {name} ===")
    model_path = cfg["model_path"]
    data_path = cfg["data_path"]
    k = cfg["k"]
    FEATURES = cfg["features"]

    # Load checkpoint
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    activation_map = {"ReLU": nn.ReLU, "Tanh": nn.Tanh, "Sigmoid": nn.Sigmoid}
    INPUT_DIM = len(FEATURES)

    class Autoencoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, latent_dim, activation_fn):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                activation_fn(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                activation_fn(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)

    ae = Autoencoder(
        INPUT_DIM,
        ckpt["best_hidden_size"],
        ckpt["best_latent_dim"],
        activation_map[ckpt["best_activation_name"]],
    )
    ae.load_state_dict(ckpt["model_state_dict"])
    ae.eval()

    # Load data and use train split (80%) to match all-in-all.py
    df = pd.read_csv(data_path)
    X_full = df[FEATURES].values
    
    # Use same split as all-in-all.py: TEST_SIZE=0.2, RANDOM_STATE=42
    X_train, _ = train_test_split(
        X_full,
        test_size=0.2,
        random_state=ckpt.get("RANDOM_SEED", 42),
        shuffle=True
    )
    
    # Fit scaler on train split only
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    with torch.no_grad():
        z = ae.encoder(torch.tensor(X_scaled, dtype=torch.float32)).numpy()

    # Fit K-means on train split only
    km = KMeans(n_clusters=k, random_state=ckpt.get("RANDOM_SEED", 42), n_init=20)
    km.fit(z)

    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(km, os.path.join(save_dir, f"{name}_kmeans_model.joblib"))
    np.save(os.path.join(save_dir, f"{name}_kmeans_centroids.npy"), km.cluster_centers_)
    joblib.dump(scaler, os.path.join(save_dir, f"{name}_scaler.joblib"))

    meta = {
        "n_clusters": k,
        "random_state": ckpt.get("RANDOM_SEED", 42),
        "latent_dim": ckpt["best_latent_dim"],
        "hidden_dim": ckpt["best_hidden_size"],
        "activation": ckpt["best_activation_name"],
    }
    with open(os.path.join(save_dir, f"{name}_kmeans_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print("Saved:")
    print(f" - {os.path.join(save_dir, f'{name}_kmeans_model.joblib')}")
    print(f" - {os.path.join(save_dir, f'{name}_kmeans_centroids.npy')}")
    print(f" - {os.path.join(save_dir, f'{name}_scaler.joblib')}")
    print(f" - {os.path.join(save_dir, f'{name}_kmeans_meta.pkl')}")


if __name__ == "__main__":
    # Set save_dir to current folder; change if you want elsewhere
    for name, cfg in DATA_CONFIG.items():
        run_one(name, cfg, save_dir=".")
