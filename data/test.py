"""
Quick check for D2-Cultural cluster occupancy using current artifacts.
Runs the D2 scaler/AE/KMeans on D2_Cultural_processed.csv and prints cluster counts.
"""

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

FEATURES = ["Depression", "Anxiety", "Stress", "Burnout"]
INPUT_DIM = 4

# Paths relative to repo root
CSV_PATH = "../D2_Cultural_processed.csv"
SCALER_PATH = "../D2-Cultural_scaler.joblib"
AE_PATH = "../D2_Cultural_model (1).pth"
KMEANS_PATH = "../D2-Cultural_kmeans_model.joblib"

activation_map = {"ReLU": nn.ReLU, "Tanh": nn.Tanh, "Sigmoid": nn.Sigmoid}


def load_autoencoder(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    class AE(nn.Module):
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

    ae = AE(
        INPUT_DIM,
        ckpt["best_hidden_size"],
        ckpt["best_latent_dim"],
        activation_map[ckpt["best_activation_name"]],
    )
    ae.load_state_dict(ckpt["model_state_dict"])
    ae.eval()
    return ae, ckpt


def main():
    df = pd.read_csv(CSV_PATH)
    X = df[FEATURES].values

    scaler = joblib.load(SCALER_PATH)
    km = joblib.load(KMEANS_PATH)
    ae, ckpt = load_autoencoder(AE_PATH)

    X_scaled = scaler.transform(X)
    with torch.no_grad():
        z = ae.encoder(torch.tensor(X_scaled, dtype=torch.float32)).numpy()

    # normalize using D2 latent stats (as in all-in-all when D2 is reference)
    z_mean = z.mean(0)
    z_std = z.std(0) + 1e-8
    z_norm = (z - z_mean) / z_std

    labels = km.predict(z_norm)
    counts = np.bincount(labels, minlength=km.n_clusters)

    print(f"Data shape: {X.shape}")
    print(f"Latent shape: {z.shape}, latent dim: {z.shape[1]}")
    print(f"K (km.n_clusters): {km.n_clusters}")
    print("Cluster counts:", counts.tolist())

    # Quick profile means to compare with expected table
    for k in range(km.n_clusters):
        mask = labels == k
        if mask.sum() == 0:
            print(f"Cluster {k}: EMPTY")
            continue
        means = X[mask].mean(axis=0)
        print(
            f"Cluster {k}: N={mask.sum()}, "
            f"Dep={means[0]:.6f}, Anx={means[1]:.6f}, Str={means[2]:.6f}, Bur={means[3]:.6f}"
        )


if __name__ == "__main__":
    main()

