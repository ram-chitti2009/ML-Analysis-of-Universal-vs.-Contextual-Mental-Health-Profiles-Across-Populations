"""
Cross-Population Alignment: Testing Profile Consistency Across Datasets
Projects every dataset into every other dataset's latent space and tests consistency
"""

import os
import pickle
from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import warnings
from scipy.stats import chi2_contingency

warnings.filterwarnings('ignore')

# Dataset and PCA artifact mappings (produced by pca_clustering.py)
DATASETS = {
    "D1-Swiss": {
        "csv": "D1_Swiss_processed_test.csv",
        "scaler": "pca_kmeans_artifacts/D1-Swiss_pca_scaler.joblib",
        "pca": "pca_kmeans_artifacts/D1-Swiss_pca_model.joblib",
        "kmeans": "pca_kmeans_artifacts/D1-Swiss_pca_kmeans.joblib",
    },
    "D2-Cultural": {
        "csv": "D2_Cultural_processed_test.csv",
        "scaler": "pca_kmeans_artifacts/D2-Cultural_pca_scaler.joblib",
        "pca": "pca_kmeans_artifacts/D2-Cultural_pca_model.joblib",
        "kmeans": "pca_kmeans_artifacts/D2-Cultural_pca_kmeans.joblib",
    },
    "D3-Academic": {
        "csv": "D3_Academic_processed_test.csv",
        "scaler": "pca_kmeans_artifacts/D3-Academic_pca_scaler.joblib",
        "pca": "pca_kmeans_artifacts/D3-Academic_pca_model.joblib",
        "kmeans": "pca_kmeans_artifacts/D3-Academic_pca_kmeans.joblib",
    },
    "D4-Tech": {
        "csv": "D4_Tech_processed_test.csv",
        "scaler": "pca_kmeans_artifacts/D4-Tech_pca_scaler.joblib",
        "pca": "pca_kmeans_artifacts/D4-Tech_pca_model.joblib",
        "kmeans": "pca_kmeans_artifacts/D4-Tech_pca_kmeans.joblib",
    },
}

feature_columns = ["Depression", "Anxiety", "Stress", "Burnout"]
INPUT_DIM = 4

# Load all datasets
print("="*80)
print("LOADING ALL DATASETS")
print("="*80)
dataset_dfs = {}
dataset_features = {}
for name, paths in DATASETS.items():
    train_df = pd.read_csv(paths["csv"])
    dataset_dfs[name] = train_df
    dataset_features[name] = train_df[feature_columns].values
    print(f"Loaded {name}: {len(train_df)} samples (pre-split training dataset)")

print(f"\nTotal datasets: {len(DATASETS)}")

# Load all PCA models, scalers, and kmeans
print("\n" + "="*80)
print("LOADING ALL PCA MODELS, SCALERS, AND K-MEANS")
print("="*80)

pcas = {}
scalers = {}
kmeans_models = {}

for name, paths in DATASETS.items():
    print(f"\nLoading {name}...")
    scaler = joblib.load(paths["scaler"])
    pca = joblib.load(paths["pca"])
    kmeans = joblib.load(paths["kmeans"])
    scalers[name] = scaler
    pcas[name] = pca
    kmeans_models[name] = kmeans
    print(f"  PCA loaded: dim={pca.n_components_}")
    print(f"  Scaler loaded")
    print(f"  K-means loaded: {kmeans.n_clusters} clusters")

print("\nAll models, scalers, and K-means loaded successfully")

# Statistical test functions
def cluster_entropy(labels, K):
    """Compute normalized entropy of cluster assignments"""
    probs = np.bincount(labels, minlength=K) / len(labels)
    probs = probs[probs > 0]
    entropy_val = -np.sum(probs * np.log(probs)) / np.log(K) if K > 1 else 0.0
    return entropy_val

def test_cluster_similarity(pop_labels, ref_labels, k, pop_name, ref_name):
    pop_counts = np.bincount(pop_labels, minlength=k)
    ref_counts = np.bincount(ref_labels, minlength=k)
    totals = pop_counts + ref_counts
    keep = totals > 0
    pop_trim = pop_counts[keep].astype(int)
    ref_trim = ref_counts[keep].astype(int)
    if len(pop_trim) < 2:
        return {'reference': ref_name, 'target': pop_name, 'chi2': np.nan, 'p_value': np.nan, 'cramers_v': np.nan, 'interpretation': 'Insufficient data'}
    contingency = np.vstack([pop_trim, ref_trim])
    try:
        chi2, p_value, dof, expected = chi2_contingency(contingency)
    except ValueError:
        return {'reference': ref_name, 'target': pop_name, 'chi2': np.nan, 'p_value': np.nan, 'cramers_v': np.nan, 'interpretation': 'chi2 error'}
    n = contingency.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1))) if chi2 > 0 else 0.0
    interpretation = 'Universal' if cramers_v < 0.3 else 'Contextual'
    print(f"Cramér's V: {cramers_v:.4f} (p={p_value:.4f}) → {interpretation}")
    return {'reference': ref_name, 'target': pop_name, 'chi2': chi2, 'p_value': p_value, 'cramers_v': cramers_v, 'interpretation': interpretation}

def compute_feature_deviation(ref_profiles, target_profiles, ref_k):
    """
    RMSD of feature profiles (Depression, Anxiety, Stress, Burnout) 
    across matching clusters.
    """
    rmsds = []
    feature_keys = ["Depression", "Anxiety", "Stress", "Burnout"]
    
    for cluster_id in range(ref_k):
        # Skip empty ref clusters
        if ref_profiles[cluster_id].get('empty', False):
            continue
        # Skip if target cluster doesn't exist
        if cluster_id not in target_profiles:
            continue
        
        ref_prof = ref_profiles[cluster_id]
        target_prof = target_profiles[cluster_id]
        
        # Build feature vectors
        ref_vec = np.array([ref_prof[f] for f in feature_keys])
        target_vec = np.array([target_prof[f] for f in feature_keys])
        
        # RMSD between the two profiles
        rmsd = np.sqrt(np.mean((ref_vec - target_vec)**2))
        rmsds.append(rmsd)
    
    return np.mean(rmsds) if rmsds else np.nan

# Main cross-population projection loop
print("\n" + "="*80)
print("CROSS-POPULATION PROJECTION ANALYSIS")
print("="*80)
print("Projecting every dataset into every other dataset's latent space")
print("="*80)

all_results = []
profile_comparisons = []

dataset_names = list(DATASETS.keys())

for ref_name in dataset_names:
    print(f"\n{'#'*80}")
    print(f"REFERENCE DATASET: {ref_name}")
    print(f"{'#'*80}")
    
    # Get reference model, scaler, kmeans
    ref_scaler = scalers[ref_name]
    ref_pca = pcas[ref_name]
    ref_kmeans = kmeans_models[ref_name]
    ref_k = ref_kmeans.n_clusters
    ref_data = dataset_features[ref_name]
    
    # Scale and project reference dataset into PCA latent space
    ref_scaled = ref_scaler.transform(ref_data)
    ref_latent = ref_pca.transform(ref_scaled)

    # Cluster reference dataset
    ref_labels = ref_kmeans.predict(ref_latent)
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

        # Convert to DataFrame for safe column access IMP - ( use column names instead of hardcoded indices as sometimes the indices are not consistent)
        ref_df = pd.DataFrame(ref_data, columns=feature_columns)
        ref_profiles[cluster_id] = {
            "Depression": ref_df.loc[mask, "Depression"].mean(),
            "Anxiety": ref_df.loc[mask, "Anxiety"].mean(),
            "Stress": ref_df.loc[mask, "Stress"].mean(),
            "Burnout": ref_df.loc[mask, "Burnout"].mean(),
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
        
        # Scale with reference scaler and project with reference PCA
        target_scaled = ref_scaler.transform(target_data)
        target_latent = ref_pca.transform(target_scaled)

        # Cluster with reference K-means
        target_labels = ref_kmeans.predict(target_latent)
        target_entropy = cluster_entropy(target_labels, ref_k)
        
        print(f"    Cluster distribution: {dict(Counter(target_labels))}")
        print(f"    Entropy: {target_entropy:.3f} (ref: {ref_entropy:.3f}, diff: {abs(target_entropy - ref_entropy):.4f})")
        
        # Compute target profile means for comparison
        target_profiles = {}
        for cluster_id in range(ref_k):
            mask = target_labels == cluster_id
            count = mask.sum()
            if count > 0:
                # Convert to DataFrame for safe column access (FIX: use column names instead of hardcoded indices)
                target_df = pd.DataFrame(target_data, columns=feature_columns)
                target_profiles[cluster_id] = {
                    "Depression": target_df.loc[mask, "Depression"].mean(),
                    "Anxiety": target_df.loc[mask, "Anxiety"].mean(),
                    "Stress": target_df.loc[mask, "Stress"].mean(),
                    "Burnout": target_df.loc[mask, "Burnout"].mean(),
                    "N": int(count)
                }
        
        # Store profile comparison data
        for cluster_id in range(ref_k):
            # Always include all clusters, even if empty in reference
            profile_comparisons.append({
                'Reference': ref_name,
                'Target': target_name,
                'Profile': f'P{cluster_id+1}',
                'Ref_Depression': ref_profiles[cluster_id]['Depression'],
                'Ref_Anxiety': ref_profiles[cluster_id]['Anxiety'],
                'Ref_Stress': ref_profiles[cluster_id]['Stress'],
                'Ref_Burnout': ref_profiles[cluster_id]['Burnout'],
                'Ref_N': ref_profiles[cluster_id]['N'],
                'Target_Depression': target_profiles.get(cluster_id, {}).get('Depression', np.nan),
                'Target_Anxiety': target_profiles.get(cluster_id, {}).get('Anxiety', np.nan),
                'Target_Stress': target_profiles.get(cluster_id, {}).get('Stress', np.nan),
                'Target_Burnout': target_profiles.get(cluster_id, {}).get('Burnout', np.nan),
                'Target_N': target_profiles.get(cluster_id, {}).get('N', 0),
                'Ref_Empty': ref_profiles[cluster_id].get('empty', False)
            })
        
        # Statistical test
        chi2_result = test_cluster_similarity(target_labels, ref_labels, ref_k, target_name, ref_name)
        print(f"    Cramér's V: {chi2_result['cramers_v']:.4f} (p={chi2_result['p_value']:.4f})")
        
        # Average RMSD of feature profiles
        rmsd = compute_feature_deviation(ref_profiles, target_profiles, ref_k)
        print(f"    Average Feature Profile RMSD: {rmsd:.4f}")
        
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
            'rmsd': rmsd,
            'target_n': len(target_data)
        })

# Summary table
print("\n" + "="*80)
print("SUMMARY: ALL CROSS-POPULATION PROJECTIONS")
print("="*80)
results_df = pd.DataFrame(all_results)
print(results_df.drop(columns=['interpretation']).to_string(index=False))

# Pivot table for easier viewing
print("\n" + "="*80)
print("CRAMÉR'S V MATRIX")
print("="*80)
cramers_pivot = results_df.pivot(index='target', columns='reference', values='cramers_v')
# Blank the diagonal (self projections are skipped)
for name in cramers_pivot.index:
    cramers_pivot.loc[name, name] = "" if name in cramers_pivot.columns else cramers_pivot.loc[name, name]
print(cramers_pivot.to_string())

print("\n" + "="*80)
print("RMSD MATRIX")
print("="*80)
rmsd_pivot = results_df.pivot(index='target', columns='reference', values='rmsd')
# Blank the diagonal for readability
for name in rmsd_pivot.index:
    rmsd_pivot.loc[name, name] = "" if name in rmsd_pivot.columns else rmsd_pivot.loc[name, name]
print(rmsd_pivot.to_string())

# Save results
results_df.to_csv("cross_population_results.csv", index=False)
print(f"\nResults saved to cross_population_results.csv")

# Save profile comparison table
if profile_comparisons:
    profile_comparison_df = pd.DataFrame(profile_comparisons)
    profile_comparison_df.to_csv("profile_comparison_table.csv", index=False)
    print(f"Profile comparison table saved to profile_comparison_table.csv")
    print(profile_comparison_df)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

