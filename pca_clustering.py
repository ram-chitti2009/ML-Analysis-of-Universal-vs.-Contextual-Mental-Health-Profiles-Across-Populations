"""
PCA + Kmeans Clustering
- 80/20 split, random_state=42
- Sweep PCA dims 1–4 and K=2..8; pick best by Silhouette
- Save scaler, PCA, KMeans, centroids, summary CSV
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)


RANDOM_SEED = 42
TEST_SIZE = 0.2
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
    train_df, _ = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_SEED, shuffle=True
    )
    x_train = train_df[FEATURES].values

    scaler = StandardScaler().fit(x_train)
    Xs = scaler.transform(x_train)

    best = None

    for dim in range(1, 5):
        pca = PCA(n_components=dim, random_state=RANDOM_SEED)
        z = pca.fit_transform(Xs)
        for k in range(2, 9):
            km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=20)
            labels = km.fit_predict(z)
            sil = silhouette_score(z, labels)
            ch = calinski_harabasz_score(z, labels)
            db = davies_bouldin_score(z, labels)
            cand = {
                "dim": dim,
                "k": k,
                "silhouette": sil,
                "calinski_harabasz": ch,
                "davies_bouldin": db,
                "pca": pca,
                "km": km,
                "labels": labels,
            }
            if best is None or sil > best["silhouette"]:
                best = cand
                print(
                    f"New best: dim={dim}, k={k}, Sil={sil:.4f}, CH={ch:.2f}, DB={db:.4f}"
                )

    print(f"\nBest configuration for {name}:")
    print(f"  PCA dim: {best['dim']}")
    print(f"  K: {best['k']}")
    print(f"  Silhouette: {best['silhouette']:.4f}")
    print(f"  Calinski-Harabasz: {best['calinski_harabasz']:.2f}")
    print(f"  Davies-Bouldin: {best['davies_bouldin']:.4f}")

    # Refit PCA and KMeans on scaled train data for saving (best already fitted)
    best_pca = best["pca"]
    best_km = best["km"]
    best_labels = best["labels"]

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
        cluster_df = train_df.loc[mask, FEATURES]
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
    
