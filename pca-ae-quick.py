"""
Quick PCA vs Autoencoder comparison using existing artifacts.
Runs across all datasets defined in ARTIFACTS.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.model_selection import train_test_split
import joblib

# Configuration
RANDOM_SEED = 42
FEATURE_COLUMNS = ["Depression", "Anxiety", "Stress", "Burnout"]
INPUT_DIM = 4
TEST_SIZE = 0.2  # match training split

# Artifact map (adjust filenames if yours differ)
ARTIFACTS = {
    "D1-Swiss": {
        "csv": "D1_Swiss_processed.csv",
        "model": "D1_Swiss_model (3).pth",
        "scaler": "D1-Swiss_scaler.joblib",
        "kmeans": "D1-Swiss_kmeans_model.joblib",
    },
    "D2-Cultural": {
        "csv": "D2_Cultural_processed.csv",
        "model": "D2_Cultural_model (2).pth",
        "scaler": "D2-Cultural_scaler.joblib",
        "kmeans": "D2-Cultural_kmeans_model.joblib",
    },
    "D3-Academic": {
        "csv": "D3_Academic_processed.csv",
        "model": "D3_Academic_model (2).pth",
        "scaler": "D3-Academic_scaler.joblib",
        "kmeans": "D3-Academic_kmeans_model.joblib",
    },
    "D4-Tech": {
        "csv": "D4_Tech_processed.csv",
        "model": "D4_Tech_model (2).pth",
        "scaler": "D4-Tech_scaler.joblib",
        "kmeans": "D4-Tech_kmeans_model.joblib",
    },
}


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, activation_function):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation_function(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            activation_function(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def fmt(x, precision=6):
    if isinstance(x, float):
        return f"{x:.{precision}f}"
    return x


def run_one(name, cfg):
    print("\n" + "=" * 80)
    print(f"{name}: PCA vs Autoencoder")
    print("=" * 80)

    try:
        df = pd.read_csv(cfg["csv"])
        train_df, _ = train_test_split(
            df,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            shuffle=True,
        )
        train_data = train_df[FEATURE_COLUMNS].values
    except Exception as e:
        print(f"  !! Failed to load data: {e}")
        return

    try:
        ckpt = torch.load(cfg["model"], map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  !! Failed to load model: {e}")
        return

    activation_map = {"ReLU": nn.ReLU, "Tanh": nn.Tanh, "Sigmoid": nn.Sigmoid}
    activation_fn = activation_map[ckpt["best_activation_name"]]

    model = Autoencoder(
        INPUT_DIM,
        ckpt["best_hidden_size"],
        ckpt["best_latent_dim"],
        activation_fn,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    try:
        scaler = joblib.load(cfg["scaler"])
        kmeans = joblib.load(cfg["kmeans"])
    except Exception as e:
        print(f"  !! Failed to load scaler/KMeans: {e}")
        return

    best_k = ckpt["best_k"]
    best_latent_dim = ckpt["best_latent_dim"]

    print(
        f"  Model: {INPUT_DIM}→{ckpt['best_hidden_size']}→{best_latent_dim}, K={best_k}"
    )
    print(f"  Train split: {len(train_data)} samples")

    # AE latent + metrics
    train_scaled = scaler.transform(train_data)
    with torch.no_grad():
        z = model.encoder(torch.tensor(train_scaled, dtype=torch.float32)).numpy()
    z_mean, z_std = z.mean(0), z.std(0)
    z_norm = (z - z_mean) / (z_std + 1e-8)
    ae_labels = kmeans.predict(z_norm)
    ae_sil = silhouette_score(z_norm, ae_labels)
    ae_ch = calinski_harabasz_score(z_norm, ae_labels)
    ae_db = davies_bouldin_score(z_norm, ae_labels)

    # PCA sweep 1-4 dims
    pca_results = []
    for pca_dim in range(1, 5):
        pca = PCA(n_components=pca_dim, random_state=RANDOM_SEED)
        p_lat = pca.fit_transform(train_scaled)
        explained_var = pca.explained_variance_ratio_.sum()

        p_km = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=20)
        p_labels = p_km.fit_predict(p_lat)

        p_sil = silhouette_score(p_lat, p_labels)
        p_ch = calinski_harabasz_score(p_lat, p_labels)
        p_db = davies_bouldin_score(p_lat, p_labels)

        pca_results.append(
            {
                "dim": pca_dim,
                "explained_var": explained_var,
                "silhouette": p_sil,
                "calinski_harabasz": p_ch,
                "davies_bouldin": p_db,
            }
        )
        print(
            f"    PCA dim={pca_dim}: Sil={p_sil:.4f}, CH={p_ch:.1f}, DB={p_db:.4f}, Var={explained_var:.4f}"
        )

    pca_df = pd.DataFrame(pca_results)
    best_idx = pca_df["silhouette"].idxmax()
    best = pca_df.iloc[best_idx]

    # Improvements (DB: lower is better)
    sil_improve = ((ae_sil - best["silhouette"]) / best["silhouette"]) * 100
    ch_improve = ((ae_ch - best["calinski_harabasz"]) / best["calinski_harabasz"]) * 100
    db_improve = ((best["davies_bouldin"] - ae_db) / best["davies_bouldin"]) * 100

    print("\n  Best PCA:")
    print(
        f"    dim={int(best['dim'])}, Sil={best['silhouette']:.4f}, "
        f"CH={best['calinski_harabasz']:.1f}, DB={best['davies_bouldin']:.4f}"
    )
    print("\n  Autoencoder:")
    print(f"    dim={best_latent_dim}, Sil={ae_sil:.4f}, CH={ae_ch:.1f}, DB={ae_db:.4f}")

    print("\n  Improvements (AE vs best PCA):")
    print(f"    Silhouette: {sil_improve:+.2f}%")
    print(f"    Calinski-Harabasz: {ch_improve:+.2f}%")
    print(f"    Davies-Bouldin: {db_improve:+.2f}% (positive = AE better)")

    # Summary row
    summary_row = {
        "dataset": name,
        "ae_latent_dim": best_latent_dim,
        "pca_dim": int(best["dim"]),
        "ae_silhouette": ae_sil,
        "pca_silhouette": best["silhouette"],
        "ae_ch": ae_ch,
        "pca_ch": best["calinski_harabasz"],
        "ae_db": ae_db,
        "pca_db": best["davies_bouldin"],
        "sil_improve_pct": sil_improve,
        "ch_improve_pct": ch_improve,
        "db_improve_pct": db_improve,
    }
    return summary_row


def main():
    all_summaries = []
    for name, cfg in ARTIFACTS.items():
        summary = run_one(name, cfg)
        if summary:
            all_summaries.append(summary)

    if all_summaries:
        print("\n" + "=" * 80)
        print("SUMMARY (AE vs Best PCA)")
        print("=" * 80)
        df = pd.DataFrame(all_summaries)
        # Order columns for readability
        df = df[
            [
                "dataset",
                "ae_latent_dim",
                "pca_dim",
                "ae_silhouette",
                "pca_silhouette",
                "sil_improve_pct",
                "ae_ch",
                "pca_ch",
                "ch_improve_pct",
                "ae_db",
                "pca_db",
                "db_improve_pct",
            ]
        ]
        # Round for printing
        print(df.round(4).to_string(index=False))
        df.to_csv("pca_vs_ae_summary.csv", index=False)
        print("\nSaved: pca_vs_ae_summary.csv")
    else:
        print("No results to summarize.")


if __name__ == "__main__":
    main()

