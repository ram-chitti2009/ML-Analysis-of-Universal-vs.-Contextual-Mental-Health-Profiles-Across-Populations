"""
Cross-Population Alignment: Testing Profile Consistency Across Datasets
Tests if profiles identified in one dataset are consistent across other datasets
"""




import os
import pickle
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import joblib
import warnings
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split

#loading the ds
DATASETS = {
    "D1-Swiss": "D1_Swiss_processed.csv",
    "D2-Cultural": "D2_Cultural_processed.csv",
    "D3-Academic": "D3_Academic_processed.csv",
    "D4-Tech": "D4_Tech_processed.csv",
}

D1_Swiss = pd.read_csv("D1_Swiss_processed.csv")
D2_Cultural = pd.read_csv("D2_Cultural_processed.csv")
D3_Academic = pd.read_csv("D3_Academic_processed.csv")
D4_Tech = pd.read_csv("D4_Tech_processed.csv")


feature_columns = ["Depression", "Anxiety", "Stress", "Burnout"]
TEST_SIZE = 0.2
RANDOM_STATE = 42

# use train split for consistency with trained artifacts
D1_Swiss, _ = train_test_split(D1_Swiss, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
D2_Cultural, _ = train_test_split(D2_Cultural, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
D3_Academic, _ = train_test_split(D3_Academic, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
D4_Tech, _ = train_test_split(D4_Tech, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)

X_Swiss = D1_Swiss[feature_columns]
X_Cultural = D2_Cultural[feature_columns]
X_Academic = D3_Academic[feature_columns]
X_Tech = D4_Tech[feature_columns]

#initialize the scaler on the swiss dataset
scaler = StandardScaler()
X_swiss_scaled = scaler.fit_transform(X_Swiss)

print(X_Swiss.head())
print(X_Cultural.head())
print(X_Academic.head())
print(X_Tech.head())
print(f"Swiss scaled shape: {X_swiss_scaled.shape}")

#load the trained autoencoder model and K-means 
print("Loading trained models...")
model_path = "D1_Swiss_model.pth"
h3_data_path = "h3_data.pkl"  # Will try to load, generate if not exists

# Load PyTorch model and reconstruct architecture
print("Loading Swiss Autoencoder model...")
try:
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Reconstruct model architecture
    import torch.nn as nn
    INPUT_DIM = 4  # Depression, Anxiety, Stress, Burnout
    
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
    
    activation_map = {'ReLU': nn.ReLU, 'Tanh': nn.Tanh, 'Sigmoid': nn.Sigmoid}
    activation_fn = activation_map[checkpoint['best_activation_name']]
    
    ae_swiss = Autoencoder(
        INPUT_DIM,
        checkpoint['best_hidden_size'],
        checkpoint['best_latent_dim'],
        activation_fn
    )
    ae_swiss.load_state_dict(checkpoint['model_state_dict'])
    ae_swiss.eval()
    print(f"Loaded Swiss Autoencoder: {INPUT_DIM} -> {checkpoint['best_hidden_size']} -> {checkpoint['best_latent_dim']}")
    print(f"  Activation: {checkpoint['best_activation_name']}, K: {checkpoint['best_k']}")
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()
    raise

# Load K-means from h3_data (or generate if not exists)
print("Loading Swiss K-means from h3_data...")
h3_data = None
if os.path.exists(h3_data_path):
    try:
        with open(h3_data_path, 'rb') as f:
            h3_data = pickle.load(f)
        print(f"✓ Loaded h3_data.pkl")
    except Exception as e:
        print(f"Warning: Could not load h3_data.pkl: {e}")

if h3_data is None or 'cluster_centroids' not in h3_data:
    print("h3_data.pkl not found or missing cluster_centroids. Generating from model...")
    print("Note: You may need to run h3_data_generation.py first, or encode Swiss data here")
    # For now, we'll need to encode Swiss data to get centroids
    # This is a placeholder - you may need to generate h3_data.pkl first
    raise ValueError("h3_data.pkl with cluster_centroids is required. Please run h3_data_generation.py first.")

from sklearn.cluster import KMeans
# Reconstruct KMeans from centroids (K-means wasn't saved separately, only centroids in h3_data)
# Need to fit KMeans first to initialize all internal attributes (_n_threads, etc.), then set centroids
km_swiss = KMeans(n_clusters=len(h3_data['cluster_centroids']), random_state=checkpoint.get('RANDOM_SEED', 42), n_init=10)
# Fit on the centroids themselves to initialize all internal attributes
km_swiss.fit(h3_data['cluster_centroids'])
# Now set the actual centroids (this preserves the initialized internal state)
km_swiss.cluster_centers_ = h3_data['cluster_centroids']
print(f"✓ Reconstructed K-means with {len(h3_data['cluster_centroids'])} clusters from h3_data.pkl")

print("loaded the trained models successfully")

#extract swiss reference profiles
print("Extracting Swiss reference profiles...")
# Use latent vectors from h3_data if available, otherwise encode
if 'latent_vectors_all' in h3_data:
    swiss_latent = h3_data['latent_vectors_all']
    print(f"✓ Using latent vectors from h3_data.pkl: {swiss_latent.shape}")
else:
    print("Encoding Swiss data through autoencoder...")
with torch.no_grad():
    X_Swiss_tensor = torch.tensor(X_swiss_scaled, dtype=torch.float32)
    swiss_latent = ae_swiss.encoder(X_Swiss_tensor).numpy()
    print(f"✓ Encoded Swiss data: {swiss_latent.shape}")

#Normalize the latent space
print("Normalizing the latent space...")
latent_swiss_mean = swiss_latent.mean(0)
latent_swiss_std = swiss_latent.std(0)
latent_swiss_normalized = (swiss_latent - latent_swiss_mean) / latent_swiss_std

print(f"Normalized Swiss reference latent vectors shape: {latent_swiss_normalized.shape}")

#Assign Swiss to clusters using Normalized Latent Vectors
swiss_labels = km_swiss.predict(latent_swiss_normalized)
k = len(np.unique(swiss_labels))

print(f"Swiss reference profiles assigned to {k} clusters: {Counter(swiss_labels)}")
print(f"Swiss reference cluster centroids: {km_swiss.cluster_centers_}")

#Compute the centroids of the swiss reference clusters
swiss_profiles = {}
X_Swiss_values = X_Swiss.values  # Convert DataFrame to numpy array

for cluster_id in range(k):
    mask = swiss_labels == cluster_id
    count = mask.sum()
    swiss_profiles[cluster_id] = {
        "Depression" : X_Swiss_values[mask, 0].mean(),
        "Anxiety" : X_Swiss_values[mask, 1].mean(),
        "Stress" : X_Swiss_values[mask, 2].mean(),
        "Burnout" : X_Swiss_values[mask, 3].mean(),
        "N" : count,
        'pct_of_total' : count / len(swiss_labels) * 100
    }
    print(f"Swiss profile {cluster_id+1}: N={count} ({(count / len(swiss_labels) * 100):.2f}%)")

# Print all profiles
for cluster_id, profile_data in swiss_profiles.items():
    print(f"Profile {cluster_id+1}: N={profile_data['N']} ({profile_data['pct_of_total']:.2f}%)")
    for feature in feature_columns:
        print(f"  {feature}: {profile_data[feature]:.2f}")

#Map other datasets to swiss reference profiles

def encode_and_cluster(autoencoder, dataset_name, feature, k_means, scaler, latent_swiss_mean, latent_swiss_std):
    """
    Encode population data and assign to swiss reference clusters
    apply scaler before encoding to ensure consistent scaling
    normalize latent space to match swiss reference
    """
    X_scaled = scaler.transform(feature)

    # Encode through Swiss autoencoder
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        latent = autoencoder.encoder(X_tensor).numpy()
    
    latent_normalized = (latent - latent_swiss_mean) / latent_swiss_std
    labels = k_means.predict(latent_normalized)
    
    print(f"Dataset {dataset_name} assigned to {len(np.unique(labels))} clusters: {Counter(labels)}")
    print(f"Dataset {dataset_name} cluster centroids: {k_means.cluster_centers_}")
    print(f"Dataset {dataset_name} reference profiles: {swiss_profiles}")
    return labels, latent_normalized, latent


latent_malaysian, latent_malaysian_normalized, latent_malaysian_latent = encode_and_cluster(ae_swiss, "Malaysian", X_Cultural, km_swiss, scaler, latent_swiss_mean, latent_swiss_std)
latent_academic, latent_academic_normalized, latent_academic_latent = encode_and_cluster(ae_swiss, "Academic", X_Academic, km_swiss, scaler, latent_swiss_mean, latent_swiss_std)
latent_tech, latent_tech_normalized, latent_tech_latent = encode_and_cluster(ae_swiss, "Tech", X_Tech, km_swiss, scaler, latent_swiss_mean, latent_swiss_std)
# Cluster Entropy: Measures how evenly distributed people are across clusters
#Measures how evenly the population is distributed across the clusters.
#Discovers : Wheather SWISS AE preserves the natural diversity of the population.
#Interpretation: If entropy stays similar to Swiss: Natural diversity is preserved,
#============================================================================

def cluster_entropy(labels, K):
    """Compute normalized entropy of cluster assignments"""
    probs = np.bincount(labels, minlength=K) / len(labels)
    probs = probs[probs > 0]
    entropy_val = -np.sum(probs * np.log(probs))/np.log(K)
    return entropy_val
entropy_malaysian = cluster_entropy(latent_malaysian, k)
entropy_academic = cluster_entropy(latent_academic, k)
entropy_tech = cluster_entropy(latent_tech, k)
entropy_swiss = cluster_entropy(swiss_labels, k)

# Calculate entropy differences for reporting (not used for classification)
entropy_diffs = {
    "Malaysian": abs(entropy_malaysian - entropy_swiss),
    "Academic": abs(entropy_academic - entropy_swiss),
    "Tech": abs(entropy_tech - entropy_swiss)
}

entropy_results = pd.DataFrame({
    "Population" : ["Malaysian", "Academic", "Tech", "Swiss"],
    "Cluster Entropy" : [entropy_malaysian, entropy_academic, entropy_tech, entropy_swiss],
    "Difference from Swiss" : [
        entropy_diffs["Malaysian"],
        entropy_diffs["Academic"],
        entropy_diffs["Tech"],
        0.0
    ]
})

print("\nCluster Entropy Results:")
print(entropy_results)

# Statistical test: Chi-square and Cramér's V for cluster distribution similarity
def test_cluster_similarity(pop_labels, swiss_labels, k, pop_name):
    """
    Test if population's cluster distribution is similar to Swiss using:
    - Chi-square test (statistical significance)
    - Cramér's V (effect size: how different are they)
    
   
    """
    pop_counts = np.bincount(pop_labels, minlength=k)
    swiss_counts = np.bincount(swiss_labels, minlength=k)
    
    # Create contingency table
    contingency = np.array([pop_counts, swiss_counts])
    
    # Chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    # Cramér's V (effect size): 0 = no association, 1 = perfect association
    # Small effect (< 0.3) = similar enough = "Universal"
    # Medium/Large effect (≥ 0.3) = too different = "Contextual"
    n = contingency.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1))) if chi2 > 0 else 0.0
    
    # Interpretation: Use effect size (Cramér's V) for "similar enough"
    # Small effect (< 0.3) = similar enough = "Universal"
    interpretation = 'Universal' if cramers_v < 0.3 else 'Contextual'
    
    return {
        'population': pop_name,
        'chi2': chi2,
        'p_value': p_value,
        'cramers_v': cramers_v,
        'interpretation': interpretation
    }

# Test each population against Swiss
chi2_results = []
chi2_results.append(test_cluster_similarity(latent_malaysian, swiss_labels, k, "Malaysian"))
chi2_results.append(test_cluster_similarity(latent_academic, swiss_labels, k, "Academic"))
chi2_results.append(test_cluster_similarity(latent_tech, swiss_labels, k, "Tech"))

chi2_df = pd.DataFrame(chi2_results)
print("\n" + "="*70)
print("STATISTICAL TEST: Chi-square & Cramér's V")
print("="*70)
print(chi2_df[['population', 'chi2', 'p_value', 'cramers_v', 'interpretation']].to_string(index=False))
print("\nInterpretation:")
print("  - Cramér's V < 0.3 (small effect) = Universal (similar enough)")
print("  - Cramér's V ≥ 0.3 (medium/large effect) = Contextual (too different)")
print("  - p-value < 0.05 = statistically significant difference")

#Stablitiy check for clustering consistency

population_data_sizes = {
    "Malaysian" : len(X_Cultural),
    "Academic" : len(X_Academic),
    "Tech" : len(X_Tech),
    "Swiss" : len(X_Swiss)
}

# Stability assessment (for reference, not used in classification)
stability_dict = {}
for pop, size in population_data_sizes.items():
    if size>=500:
        stability_dict[pop] = "Stable"
    elif size>=200:
        stability_dict[pop] = "Stable"
    elif size>=100:
        stability_dict[pop] = "Moderate"
    else:
        stability_dict[pop] = "Unstable"


#Pre-feature deviation check
#How much symptom severity differs within the same clusters across populations.
# Discovers : Wheather the same profile means the same thing across populations.
# Interpretatio : Low Deviation = Universal meaning, High Deviation = Contextual meaning.
#This is reported for the context, only the cramers v is used for classification.
def compute_feature_deviation(X_other, labels_other, swiss_profiles, feature_columns):
    """Compute feature deviation from swiss reference profiles"""
    deviations = {}
    X_other_values = X_other.values if isinstance(X_other, pd.DataFrame) else X_other

    for k in swiss_profiles.keys():
        mask = labels_other == k
        if mask.sum() == 0:
            deviations[k] = {feature: np.nan for feature in feature_columns}
        else:
            X_cluster = X_other_values[mask]
            deviations[k] = {}
            for i, feat in enumerate(feature_columns):
                ref_value = swiss_profiles[k][feat]
                cluster_value = X_cluster[:, i].mean()
                deviation = np.abs(cluster_value - ref_value)**2
                deviations[k][feat] = deviation
    return deviations

deviation_malaysian = compute_feature_deviation(X_Cultural, latent_malaysian, swiss_profiles, feature_columns)
deviation_academic = compute_feature_deviation(X_Academic, latent_academic, swiss_profiles, feature_columns)
deviation_tech = compute_feature_deviation(X_Tech, latent_tech, swiss_profiles, feature_columns)

all_deviations = {
    "Malaysian" : deviation_malaysian,
    "Academic" : deviation_academic,
    "Tech" : deviation_tech,

}
print("\nFeature Deviation Results:")
print(all_deviations)

#Post-feature deviation check

deviation_table = []

for pop, deviations in all_deviations.items():
    for feature in feature_columns:
        vals = [deviations[k].get(feature, np.nan) for k in deviations.keys()]
        if vals:
            mean_dev = np.mean(vals)
            deviation_table.append({
                "Population" : pop,
                "Feature" : feature,
                "Mean Deviation" : mean_dev
            })
deviation_df = pd.DataFrame(deviation_table)
if not deviation_df.empty:
    dev_pivot = deviation_df.pivot(index="Feature", columns="Population", values="Mean Deviation")
    print("\nPost-Feature Deviation Results:")
    print(dev_pivot)
else:
    print("\nPost-Feature Deviation Results: No data available")

#cluster usage distribution analysis
print("\nCluster Usage Distribution Analysis:")
print("="*70)


#summary and report

print("\nSummary and Report:")
print("="*70)

print("finding the consistency of the profiles across the datasets")

for pop_name in ["Malaysian", "Academic", "Tech"]:

    print(f"\nAnalyzing {pop_name} dataset:")
    if pop_name == "Malaysian":
        entropy_val = entropy_malaysian
        deviation_val = deviation_malaysian
        chi2_result = chi2_results[0]  # Malaysian
        n=len(X_Cultural)
    elif pop_name == "Academic":
        entropy_val = entropy_academic
        deviation_val = deviation_academic
        chi2_result = chi2_results[1]  # Academic
        n=len(X_Academic)
    elif pop_name == "Tech":
        entropy_val = entropy_tech
        deviation_val = deviation_tech
        chi2_result = chi2_results[2]  # Tech
        n=len(X_Tech)
    else:
        print(f"Invalid population name: {pop_name}")
        continue

    # Calculate mean feature deviation across all clusters and features
    all_feat_deviations = []
    for k in deviation_val.keys():
        for feature, deviation in deviation_val[k].items():
            if not np.isnan(deviation):
                all_feat_deviations.append(deviation)
    mean_feature_deviation = np.mean(all_feat_deviations) if all_feat_deviations else 0.0

    # Calculate entropy difference
    entropy_diff = abs(entropy_val - entropy_swiss)

    print(f"  - Dataset size: {n} participants")
    print(f"  - Cluster Entropy: {entropy_val:.3f} (Swiss: {entropy_swiss:.3f}, diff: {entropy_diff:.4f})")
    print(f"  - Mean Feature Deviation: {mean_feature_deviation:.4f}")
    print(f"  - Cramér's V: {chi2_result['cramers_v']:.4f} (p={chi2_result['p_value']:.4f})")

    stability_note = "Stable" if n>=500 else "Moderate" if n>=200 else "Unstable"
    print(f"  - Stability: {stability_note} (N={n})")

    print(f"n={n} - entropy={entropy_val:.3f} (diff: {entropy_diff:.4f}) - mean_dev={mean_feature_deviation:.4f} - stability={stability_note}")
    
    # PRIMARY CLASSIFICATION: Based on Cramér's V (empirically justified - Cohen's effect size)
    # Cramér's V < 0.3 = small effect = "Universal" (cluster structure similar)
    # Cramér's V ≥ 0.3 = medium/large effect = "Contextual" (cluster structure different)
    #We are using cohens thing to interpret the effect size.
    #Discovers : Wheather the cluster structure is similar to the swiss structure.

    if chi2_result['cramers_v'] < 0.3:
        interpretation = "Universal"
        reason = f"Cluster structure similar (Cramér's V = {chi2_result['cramers_v']:.4f} < 0.3, small effect)"
    else:
        interpretation = "Contextual"
        reason = f"Cluster structure different (Cramér's V = {chi2_result['cramers_v']:.4f} ≥ 0.3, medium/large effect)"
    
    print(f"\nClassification: {interpretation}")
    print(f"  Primary reason: {reason}")
    print(f"  Statistical test: p-value = {chi2_result['p_value']:.4f} {'(significant)' if chi2_result['p_value'] < 0.05 else '(not significant)'}")
    
    # Context: Feature deviations (descriptive, not used for classification)
    # Compare to other populations for context
    all_mean_deviations = {
        "Malaysian": np.mean([d for k in deviation_malaysian.keys() 
                              for d in deviation_malaysian[k].values() if not np.isnan(d)]),
        "Academic": np.mean([d for k in deviation_academic.keys() 
                             for d in deviation_academic[k].values() if not np.isnan(d)]),
        "Tech": np.mean([d for k in deviation_tech.keys() 
                        for d in deviation_tech[k].values() if not np.isnan(d)])
    }
    
    other_deviations = [v for k, v in all_mean_deviations.items() if k != pop_name]
    if other_deviations:
        if mean_feature_deviation < min(other_deviations) * 1.5:
            dev_context = "Similar to or lower than other populations"
        elif mean_feature_deviation > max(other_deviations) * 1.5:
            dev_context = "Much higher than other populations"
        else:
            dev_context = "Within range of other populations"
    else:
        dev_context = "No comparison available"
    
    print(f"\n  Context - Feature Deviations:")
    print(f"    Mean deviation: {mean_feature_deviation:.4f}")
    print(f"    {dev_context}")
    print(f"    Note: Higher deviations indicate different symptom levels within clusters")

    # Collect feature deviations with feature names
    all_feat_deviations = []
    for k in deviation_val.keys():
        for feature, deviation in deviation_val[k].items():
            if not np.isnan(deviation):
                all_feat_deviations.append((feature, deviation))
    
    if all_feat_deviations:
        all_feat_deviations.sort(key=lambda x: x[1], reverse=True)
        print("\n  Top feature deviations (highest to lowest):")
        for feat_name, feat_dev in all_feat_deviations[:3]:
            print(f"    - {feat_name}: {feat_dev:.4f}")

print(f"  - Cluster Entropy: {entropy_val:.3f} (Swiss: {entropy_swiss:.3f}, diff: {entropy_diff:.4f})")
print(f"    * Structure preservation: Similar = preserved, Large drop = may be lost")
print(f"  - Mean Feature Deviation: {mean_feature_deviation:.4f}")
print(f"    * Severity differences: Low = universal meaning, High = contextual meaning")
print(f"  - Cramér's V: {chi2_result['cramers_v']:.4f} (p={chi2_result['p_value']:.4f})")
print(f"    * Distribution similarity: < 0.3 = Universal, >= 0.3 = Contextual (classification)")