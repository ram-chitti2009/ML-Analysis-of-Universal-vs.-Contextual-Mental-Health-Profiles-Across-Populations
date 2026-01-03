import numpy as np
import pandas as pd
import joblib


# Dataset and PCA mappings
DATASETS = {
    "D1-Swiss": {
        "csv": "D1_Swiss_processed_test.csv",
        "scaler": "pca_kmeans_artifacts/D1-Swiss_pca_scaler.joblib",
        "pca": "pca_kmeans_artifacts/D1-Swiss_pca_model.joblib",
    },
    "D2-Cultural": {
        "csv": "D2_Cultural_processed_test.csv",
        "scaler": "pca_kmeans_artifacts/D2-Cultural_pca_scaler.joblib",
        "pca": "pca_kmeans_artifacts/D2-Cultural_pca_model.joblib",
    },
    "D3-Academic": {
        "csv": "D3_Academic_processed_test.csv",
        "scaler": "pca_kmeans_artifacts/D3-Academic_pca_scaler.joblib",
        "pca": "pca_kmeans_artifacts/D3-Academic_pca_model.joblib",
    },
    "D4-Tech": {
        "csv": "D4_Tech_processed_test.csv",
        "scaler": "pca_kmeans_artifacts/D4-Tech_pca_scaler.joblib",
        "pca": "pca_kmeans_artifacts/D4-Tech_pca_model.joblib",
    },
}

feature_columns = ["Depression", "Anxiety", "Stress", "Burnout"]


print("=== PCA Reconstruction Error Calculation ===")

for name, paths in DATASETS.items():
    print(f"\n--- Processing {name} ---")
    df = pd.read_csv(paths["csv"])
    X = df[feature_columns].values

    # Load scaler and PCA model
    scaler = joblib.load(paths["scaler"])
    pca = joblib.load(paths["pca"])

    # Scale data
    Xs_test = scaler.transform(X)

    # PCA transformation and reconstruction
    Z_test = pca.transform(Xs_test)
    Xs_recon_test = pca.inverse_transform(Z_test)

    # Calculate reconstruction loss (MSE)
    recon_loss_test = np.mean((Xs_test - Xs_recon_test) ** 2)

    print(f"Test Reconstruction Loss (MSE): {recon_loss_test:.6f}")
    print(f"Explained Variance Ratio: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"Number of Components: {pca.n_components_}")