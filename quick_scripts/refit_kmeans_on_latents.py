import os
import shutil
import pickle
import joblib
import numpy as np
import pandas as pd
import torch
from torch.serialization import safe_globals
from sklearn.cluster import KMeans

# Minimal AE-only refit script — uses provided k per dataset
DATASETS = {
    "D1-Swiss": {"csv": "D1_Swiss_processed_train.csv", "model": "D1_Swiss_model (4).pth", "scaler": "D1-Swiss_scaler.joblib", "kmeans": "D1-Swiss_kmeans_model.joblib", "kmeans_meta": "D1-Swiss_kmeans_meta.pkl"},
    "D2-Cultural": {"csv": "D2_Cultural_processed_train.csv", "model": "D2_Cultural_model (3).pth", "scaler": "D2-Cultural_scaler.joblib", "kmeans": "D2-Cultural_kmeans_model.joblib", "kmeans_meta": "D2-Cultural_kmeans_meta.pkl"},
    "D3-Academic": {"csv": "D3_Academic_processed_train.csv", "model": "D3_Academic_model (3).pth", "scaler": "D3-Academic_scaler.joblib", "kmeans": "D3-Academic_kmeans_model.joblib", "kmeans_meta": "D3-Academic_kmeans_meta.pkl"},
    "D4-Tech": {"csv": "D4_Tech_processed_train.csv", "model": "D4_Tech_model (3).pth", "scaler": "D4-Tech_scaler.joblib", "kmeans": "D4-Tech_kmeans_model.joblib", "kmeans_meta": "D4-Tech_kmeans_meta.pkl"},
}

AE_K_OVERRIDE = {
    "D1-Swiss": 2,
    "D2-Cultural": 6,
    "D3-Academic": 2,
    "D4-Tech": 3,
}

N_INIT = 50
RANDOM_STATE = 42
buffer = 1e-8
INPUT_COLS = ["Depression", "Anxiety", "Stress", "Burnout"]


class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, activation_function=torch.nn.ReLU):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            activation_function(),
            torch.nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            activation_function(),
            torch.nn.Linear(hidden_dim, input_dim)
        )


def _ensure_dir_for(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _backup(path):
    if os.path.exists(path):
        bak = path + ".bak"
        try:
            shutil.copyfile(path, bak)
            print(f"  Backed up {path} -> {bak}")
        except Exception as e:
            print(f"  Warning: failed to back up {path}: {e}")


def refit_ae_for(name, paths, k_override):
    if not (os.path.exists(paths["csv"]) and os.path.exists(paths["model"]) and os.path.exists(paths["scaler"])):
        print("  Missing required files for", name, "— skipping")
        return

    df = pd.read_csv(paths["csv"]) 
    X = df[INPUT_COLS].values

    try:
        with safe_globals([np._core.multiarray._reconstruct]):
            ckpt = torch.load(paths["model"], map_location="cpu")
    except Exception as e:
        ckpt = torch.load(paths["model"], map_location="cpu", weights_only=False)
    activation_map = {"ReLU": torch.nn.ReLU, "Tanh": torch.nn.Tanh, "Sigmoid": torch.nn.Sigmoid}
    act_name = ckpt.get("best_activation_name", "ReLU")
    act = activation_map.get(act_name, torch.nn.ReLU)

    ae = Autoencoder(input_dim=X.shape[1], hidden_dim=int(ckpt.get("best_hidden_size", 16)), latent_dim=int(ckpt.get("best_latent_dim", 3)), activation_function=act)
    try:
        ae.load_state_dict(ckpt["model_state_dict"])
    except Exception:
        if "state_dict" in ckpt:
            ae.load_state_dict(ckpt["state_dict"])
        else:
            raise
    ae.eval()

    scaler = joblib.load(paths["scaler"]) 
    X_scaled = scaler.transform(X)

    with torch.no_grad():
        lat = ae.encoder(torch.tensor(X_scaled, dtype=torch.float32)).numpy()

    lat_mean = lat.mean(0)
    lat_std = np.maximum(lat.std(0), buffer)
    lat_norm = (lat - lat_mean) / lat_std

    # backup existing kmeans
    _backup(paths["kmeans"])

    # fit kmeans on normalized latents (use sklearn default init='k-means++')
    kmeans = KMeans(n_clusters=int(k_override), n_init=N_INIT, random_state=RANDOM_STATE)
    kmeans.fit(lat_norm)

    # save
    _ensure_dir_for(paths["kmeans"])
    joblib.dump(kmeans, paths["kmeans"])

    meta = {
        "n_clusters": int(kmeans.n_clusters),
        "random_state": RANDOM_STATE,
        "fit_on_normalized_latents": True,
        "latent_mean": lat_mean.tolist(),
        "latent_std": lat_std.tolist(),
        "latent_dim": int(lat.shape[1]),
    }
    with open(paths["kmeans_meta"], "wb") as f:
        pickle.dump(meta, f)

    centroids_path = paths["kmeans"].replace("_kmeans_model.joblib", "_kmeans_centroids.npy")
    np.save(centroids_path, kmeans.cluster_centers_)

    labels = kmeans.predict(lat_norm)
    counts = dict(zip(*np.unique(labels, return_counts=True)))
    print(f"  Done — centroids {kmeans.cluster_centers_.shape}, cluster counts: {counts}")

    # Print cluster characteristics
    print(f"Cluster characteristics for {name}:")
    for cluster_id in range(kmeans.n_clusters):
        cluster_data = df[labels == cluster_id]
        mean_values = cluster_data[INPUT_COLS].mean()
        print(f"Cluster {cluster_id}:")
        print(mean_values)
        print(f"Count: {len(cluster_data)}\n")


if __name__ == "__main__":
    for name, paths in DATASETS.items():
        k = AE_K_OVERRIDE.get(name)
        if k is None:
            print(f"Skipping {name}: no override k provided")
            continue
        try:
            refit_ae_for(name, paths, k)
        except Exception as e:
            print(f"Refit failed for {name}: {e}")

