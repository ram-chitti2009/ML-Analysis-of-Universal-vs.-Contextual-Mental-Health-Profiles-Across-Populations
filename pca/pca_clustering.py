"""
PCA + Kmeans Clustering
- 10-fold CV on full dataset for hyperparameter selection (PCA dims 1–4, K=2..8)
- Pick best by average Silhouette across folds
- Save scaler, PCA, KMeans, centroids, summary CSV
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


RANDOM_SEED = 42
FEATURES = ["Depression", "Anxiety", "Stress", "Burnout"]
DATASETS = {
    "D1-Swiss": {"csv": "D1_Swiss_processed.csv"},
    "D2-Cultural": {"csv": "D2_Cultural_processed.csv"},
    "D3-Academic": {"csv": "D3_Academic_processed.csv"},
    "D4-Tech": {"csv": "D4_Tech_processed.csv"},
}

OUTPUT_DIR = Path("pca_kmeans_artifacts")
OUTPUT_DIR.mkdir(exist_ok=True)

def run_dataset(name, config):
    print(f"\n=== Processing {name} ===")
    df = pd.read_csv(config["csv"])
    x_train = df[FEATURES].values

    scaler = StandardScaler().fit(x_train)
    Xs = scaler.transform(x_train)

    best = None
    kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

    for dim in range(1, 5):
        pca = PCA(n_components=dim, random_state=RANDOM_SEED)
        z = pca.fit_transform(Xs)
        X_recon = pca.inverse_transform(z)
        recon_loss = np.mean((Xs - X_recon)**2)
        for k in range(2, 9):
            sil_scores = []
            ch_scores = []
            db_scores = []
            for train_idx, val_idx in kf.split(z):
                z_train = z[train_idx]
                z_val = z[val_idx]
                km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=20)
                km.fit(z_train)
                labels_val = km.predict(z_val)
                if len(np.unique(labels_val)) > 1: # At least 2 clusters
                    sil_scores.append(silhouette_score(z_val, labels_val))
                    ch_scores.append(calinski_harabasz_score(z_val, labels_val))
                    db_scores.append(davies_bouldin_score(z_val, labels_val))
            if sil_scores:
                avg_sil = np.mean(sil_scores)
                avg_ch = np.mean(ch_scores)
                avg_db = np.mean(db_scores)
                cand = {
                    "dim": dim,
                    "k": k,
                    "silhouette": avg_sil,
                    "calinski_harabasz": avg_ch,
                    "davies_bouldin": avg_db,
                    "recon_loss": recon_loss,
                    "pca": pca,
                }
                if best is None or avg_sil > best["silhouette"]:
                    best = cand
                    print(
                        f"New best: dim={dim}, k={k}, Avg Sil={avg_sil:.4f}, Avg CH={avg_ch:.2f}, Avg DB={avg_db:.4f}, Recon Loss={recon_loss:.6f}"
                    )

    print(f"\nBest configuration for {name}:")
    print(f"  PCA dim: {best['dim']}")
    print(f"  K: {best['k']}")
    print(f"  Silhouette: {best['silhouette']:.4f}")
    print(f"  Calinski-Harabasz: {best['calinski_harabasz']:.2f}")
    print(f"  Davies-Bouldin: {best['davies_bouldin']:.4f}")

    # Refit PCA and KMeans on scaled train data for saving
    best_pca = best["pca"]
    z_full = best_pca.transform(Xs)  # Already fitted
    best_km = KMeans(n_clusters=best["k"], random_state=RANDOM_SEED, n_init=20)
    best_labels = best_km.fit_predict(z_full)

    # Save artifacts
    prefix = OUTPUT_DIR / f"{name}_pca"
    joblib.dump(scaler, OUTPUT_DIR / f"{name}_pca_scaler.joblib")
    joblib.dump(best_pca, OUTPUT_DIR / f"{name}_pca_model.joblib")
    joblib.dump(best_km, OUTPUT_DIR / f"{name}_pca_kmeans.joblib")
    np.save(OUTPUT_DIR / f"{name}_pca_centroids.npy", best_km.cluster_centers_)
    print(f"Saved scaler, PCA, KMeans, centroids to {OUTPUT_DIR}/")

    # Cluster profile means in original feature space (train split)
    profile_rows = []
    for cid in range(best["k"]):
        mask = best_labels == cid
        n = int(mask.sum())
        if n == 0:
            continue
        cluster_df = df.loc[mask, FEATURES]
        means = cluster_df.mean()
        profile_rows.append(
            {
                "Cluster": cid,
                "N": n,
                **{feat: means[feat] for feat in FEATURES},
            }
        )
    profiles_df = pd.DataFrame(profile_rows)
    profiles_path = OUTPUT_DIR / f"{name}_pca_profiles.csv"
    profiles_df.to_csv(profiles_path, index=False)
    print(f"Saved cluster profiles to {profiles_path}")

    return {
        "dataset": name,
        "pca_dim": best["dim"],
        "k": best["k"],
        "silhouette": best["silhouette"],
        "calinski_harabasz": best["calinski_harabasz"],
        "davies_bouldin": best["davies_bouldin"],
    }
def main():
    rows = []
    for name, config in DATASETS.items():
        row = run_dataset(name, config)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary_path = OUTPUT_DIR / "pca_kmeans_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\nOverall summary:")
    print(summary.round(4).to_string(index=False))
    print(f"\nSaved overall summary to {summary_path}")

if __name__ == "__main__":
    main()
    
