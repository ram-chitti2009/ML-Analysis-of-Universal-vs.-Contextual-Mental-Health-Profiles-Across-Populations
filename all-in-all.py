"""
Cross-Population Alignment: Testing Profile Consistency Across Datasets
Projects every dataset into every other dataset's latent space and tests consistency
"""

import os
import pickle
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import seaborn as sns
import joblib
import warnings
from scipy.stats import chi2_contingency

warnings.filterwarnings('ignore')

# Dataset and model file mappings
DATASETS = {
    "D1-Swiss": {
        "csv": "D1_Swiss_processed.csv",
        "model": "D1_Swiss_model.pth",
        "scaler": "D1-Swiss_scaler.joblib",
        "kmeans": "D1-Swiss_kmeans_model.joblib",
        "kmeans_centroids": "D1-Swiss_kmeans_centroids.npy",
        "kmeans_meta": "D1-Swiss_kmeans_meta.pkl"
    },
    "D2-Cultural": {
        "csv": "D2_Cultural_processed.csv",
        "model": "D2_Cultural_model (1).pth",
        "scaler": "D2-Cultural_scaler.joblib",
        "kmeans": "D2-Cultural_kmeans_model.joblib",
        "kmeans_centroids": "D2-Cultural_kmeans_centroids.npy",
        "kmeans_meta": "D2-Cultural_kmeans_meta.pkl"
    },
    "D3-Academic": {
        "csv": "D3_Academic_processed.csv",
        "model": "D3_Academic_model (1).pth",
        "scaler": "D3-Academic_scaler.joblib",
        "kmeans": "D3-Academic_kmeans_model.joblib",
        "kmeans_centroids": "D3-Academic_kmeans_centroids.npy",
        "kmeans_meta": "D3-Academic_kmeans_meta.pkl"
    },
    "D4-Tech": {
        "csv": "D4_Tech_processed.csv",
        "model": "D4_Tech_model.pth",
        "scaler": "D4-Tech_scaler.joblib",
        "kmeans": "D4-Tech_kmeans_model.joblib",
        "kmeans_centroids": "D4-Tech_kmeans_centroids.npy",
        "kmeans_meta": "D4-Tech_kmeans_meta.pkl"
    }
}

feature_columns = ["Depression", "Anxiety", "Stress", "Burnout"]
INPUT_DIM = 4
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Load all datasets
print("="*80)
print("LOADING ALL DATASETS")
print("="*80)
dataset_dfs = {}
dataset_features = {}
for name, paths in DATASETS.items():
    full_df = pd.read_csv(paths["csv"])
    train_df, _ = train_test_split(
        full_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    dataset_dfs[name] = train_df
    dataset_features[name] = train_df[feature_columns].values
    print(f"✓ Loaded {name}: {len(train_df)} samples (train split from {len(full_df)})")

print(f"\nTotal datasets: {len(DATASETS)}")

# Define Autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, activation_function):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation_function(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            activation_function(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Load all models, scalers, and kmeans
print("\n" + "="*80)
print("LOADING ALL MODELS, SCALERS, AND K-MEANS")
print("="*80)

models = {}
scalers = {}
kmeans_models = {}
activation_map = {'ReLU': nn.ReLU, 'Tanh': nn.Tanh, 'Sigmoid': nn.Sigmoid}

for name, paths in DATASETS.items():
    print(f"\nLoading {name}...")
    
    # Load autoencoder
    checkpoint = torch.load(paths["model"], map_location='cpu', weights_only=False)
    activation_fn = activation_map[checkpoint['best_activation_name']]
    
    model = Autoencoder(
        INPUT_DIM,
        checkpoint['best_hidden_size'],
        checkpoint['best_latent_dim'],
        activation_fn
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    models[name] = {
        'model': model,
        'hidden_dim': checkpoint['best_hidden_size'],
        'latent_dim': checkpoint['best_latent_dim'],
        'activation': checkpoint['best_activation_name'],
        'k': checkpoint['best_k']
    }
    print(f"  ✓ Autoencoder: {INPUT_DIM}→{checkpoint['best_hidden_size']}→{checkpoint['best_latent_dim']}, K={checkpoint['best_k']}")
    
    # Load scaler
    scaler = joblib.load(paths["scaler"])
    scalers[name] = scaler
    print(f"  ✓ Scaler loaded")
    
    # Load KMeans
    kmeans = joblib.load(paths["kmeans"])
    kmeans_models[name] = kmeans
    print(f"  ✓ K-means loaded: {kmeans.n_clusters} clusters")

print("\n✓ All models, scalers, and K-means loaded successfully")

# Statistical test functions
def cluster_entropy(labels, K):
    """Compute normalized entropy of cluster assignments"""
    probs = np.bincount(labels, minlength=K) / len(labels)
    probs = probs[probs > 0]
    entropy_val = -np.sum(probs * np.log(probs)) / np.log(K) if K > 1 else 0.0
    return entropy_val

def test_cluster_similarity(pop_labels, ref_labels, k, pop_name, ref_name):
    """Test if population's cluster distribution is similar to reference using Chi-square and Cramér's V"""
    pop_counts = np.bincount(pop_labels, minlength=k)
    ref_counts = np.bincount(ref_labels, minlength=k)
    
    contingency = np.array([pop_counts, ref_counts])
    try:
        chi2, p_value, dof, expected = chi2_contingency(contingency)
    except ValueError:
        # If expected frequencies contain zeros, chi2_contingency raises; safeguard for sparse/empty clusters
        chi2 = 0.0
        p_value = 1.0
        dof = 0
        expected = contingency.astype(float)
    
    n = contingency.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1))) if chi2 > 0 else 0.0
    
    interpretation = 'Universal' if cramers_v < 0.3 else 'Contextual'
    
    return {
        'reference': ref_name,
        'target': pop_name,
        'chi2': chi2,
        'p_value': p_value,
        'cramers_v': cramers_v,
        'interpretation': interpretation
    }

def compute_feature_deviation(X_target, labels_target, ref_profiles, feature_columns):
    """Compute feature deviation from reference profiles"""
    deviations = {}
    X_target_values = X_target if isinstance(X_target, np.ndarray) else X_target.values
    
    for k in ref_profiles.keys():
        mask = labels_target == k
        # Skip empty reference clusters to avoid None/NaN math
        if ref_profiles[k].get("empty"):
            deviations[k] = {feature: np.nan for feature in feature_columns}
            continue

        if mask.sum() == 0:
            deviations[k] = {feature: np.nan for feature in feature_columns}
        else:
            X_cluster = X_target_values[mask]
            deviations[k] = {}
            for i, feat in enumerate(feature_columns):
                ref_value = ref_profiles[k][feat]
                cluster_value = X_cluster[:, i].mean()
                deviation = np.abs(cluster_value - ref_value) ** 2
                deviations[k][feat] = deviation
    return deviations

# Main cross-population projection loop
print("\n" + "="*80)
print("CROSS-POPULATION PROJECTION ANALYSIS")
print("="*80)
print("Projecting every dataset into every other dataset's latent space")
print("="*80)

all_results = []

dataset_names = list(DATASETS.keys())

for ref_name in dataset_names:
    print(f"\n{'#'*80}")
    print(f"REFERENCE DATASET: {ref_name}")
    print(f"{'#'*80}")
    
    # Get reference model, scaler, kmeans
    ref_model = models[ref_name]['model']
    ref_scaler = scalers[ref_name]
    ref_kmeans = kmeans_models[ref_name]
    ref_k = models[ref_name]['k']
    ref_data = dataset_features[ref_name]
    
    # Scale and encode reference dataset
    ref_scaled = ref_scaler.transform(ref_data)
    with torch.no_grad():
        ref_tensor = torch.tensor(ref_scaled, dtype=torch.float32)
        ref_latent = ref_model.encoder(ref_tensor).numpy()
    
    # Normalize latent space
    ref_latent_mean = ref_latent.mean(0)
    ref_latent_std = ref_latent.std(0)
    ref_latent_normalized = (ref_latent - ref_latent_mean) / (ref_latent_std + 1e-8)
    
    # Cluster reference dataset
    ref_labels = ref_kmeans.predict(ref_latent_normalized)
    ref_entropy = cluster_entropy(ref_labels, ref_k)
    
    # Extract reference profiles
    ref_profiles = {}
    for cluster_id in range(ref_k):
        mask = ref_labels == cluster_id
        count = mask.sum()

        # If a cluster has no members, avoid NaNs and surface an explicit flag
        if count == 0:
            ref_profiles[cluster_id] = {
                "Depression": None,
                "Anxiety": None,
                "Stress": None,
                "Burnout": None,
                "N": 0,
                'pct_of_total': 0.0,
                'empty': True
            }
            continue

        ref_profiles[cluster_id] = {
            "Depression": ref_data[mask, 0].mean(),
            "Anxiety": ref_data[mask, 1].mean(),
            "Stress": ref_data[mask, 2].mean(),
            "Burnout": ref_data[mask, 3].mean(),
            "N": int(count),
            'pct_of_total': float(count / len(ref_labels) * 100),
            'empty': False
        }
    
    print(f"\n{ref_name} Reference Profiles (K={ref_k}):")
    for cluster_id, profile in ref_profiles.items():
        if profile.get('empty'):
            print(f"  Profile {cluster_id+1}: N=0 (0.0%) - EMPTY CLUSTER")
            continue
        print(f"  Profile {cluster_id+1}: N={profile['N']} ({profile['pct_of_total']:.1f}%) - "
              f"Dep={profile['Depression']:.2f}, Anx={profile['Anxiety']:.2f}, "
              f"Str={profile['Stress']:.2f}, Bur={profile['Burnout']:.2f}")
    
    # Project all other datasets into this reference space
    for target_name in dataset_names:
        if target_name == ref_name:
            print(f"\n  → Skipping {target_name} (same as reference)")
            continue
        
        print(f"\n  → Projecting {target_name} into {ref_name} latent space...")
        
        target_data = dataset_features[target_name]
        
        # Scale with reference scaler
        target_scaled = ref_scaler.transform(target_data)
        
        # Encode with reference autoencoder
        with torch.no_grad():
            target_tensor = torch.tensor(target_scaled, dtype=torch.float32)
            target_latent = ref_model.encoder(target_tensor).numpy()
        
        # Normalize with reference statistics
        target_latent_normalized = (target_latent - ref_latent_mean) / (ref_latent_std + 1e-8)
        
        # Cluster with reference K-means
        target_labels = ref_kmeans.predict(target_latent_normalized)
        target_entropy = cluster_entropy(target_labels, ref_k)
        
        print(f"    Cluster distribution: {dict(Counter(target_labels))}")
        print(f"    Entropy: {target_entropy:.3f} (ref: {ref_entropy:.3f}, diff: {abs(target_entropy - ref_entropy):.4f})")
        
        # Statistical test
        chi2_result = test_cluster_similarity(target_labels, ref_labels, ref_k, target_name, ref_name)
        print(f"    Cramér's V: {chi2_result['cramers_v']:.4f} (p={chi2_result['p_value']:.4f}) → {chi2_result['interpretation']}")
        
        # Feature deviation
        deviations = compute_feature_deviation(target_data, target_labels, ref_profiles, feature_columns)
        all_feat_deviations = [d for k in deviations.keys() for d in deviations[k].values() if not np.isnan(d)]
        mean_deviation = np.mean(all_feat_deviations) if all_feat_deviations else 0.0
        print(f"    Mean feature deviation: {mean_deviation:.4f}")
        
        # Store results
        all_results.append({
            'reference': ref_name,
            'target': target_name,
            'ref_k': ref_k,
            'ref_entropy': ref_entropy,
            'target_entropy': target_entropy,
            'entropy_diff': abs(target_entropy - ref_entropy),
            'cramers_v': chi2_result['cramers_v'],
            'p_value': chi2_result['p_value'],
            'interpretation': chi2_result['interpretation'],
            'mean_deviation': mean_deviation,
            'target_n': len(target_data)
        })

# Summary table
print("\n" + "="*80)
print("SUMMARY: ALL CROSS-POPULATION PROJECTIONS")
print("="*80)
results_df = pd.DataFrame(all_results)
print(results_df.to_string(index=False))

# Pivot table for easier viewing
print("\n" + "="*80)
print("CRAMÉR'S V MATRIX (< 0.3 = Universal, ≥ 0.3 = Contextual)")
print("="*80)
cramers_pivot = results_df.pivot(index='target', columns='reference', values='cramers_v')
# Blank the diagonal (self projections are skipped)
for name in cramers_pivot.index:
    if name in cramers_pivot.columns:
        cramers_pivot.loc[name, name] = ""
print(cramers_pivot.to_string())

print("\n" + "="*80)
print("INTERPRETATION MATRIX")
print("="*80)
interp_pivot = results_df.pivot(index='target', columns='reference', values='interpretation')
# Blank the diagonal for readability
for name in interp_pivot.index:
    if name in interp_pivot.columns:
        interp_pivot.loc[name, name] = ""
print(interp_pivot.to_string())

# Save results
results_df.to_csv("cross_population_results.csv", index=False)
print(f"\n✓ Results saved to cross_population_results.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)