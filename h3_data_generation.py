"""
Regenerate H3 data from saved D1-Swiss model
Run this in Colab to create h3_data.pkl from my saved .pth file
"""
from google.colab import drive
drive.mount('/content/drive')
import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader, TensorDataset


RANDOM_SEED = 42
FEATURE_COLUMNS = ["Depression", "Anxiety", "Stress", "Burnout"]
INPUT_DIM = 4

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

MODEL_PATH = "/content/drive/MyDrive/CAPTURE/D1_Swiss_model.pth"  # Adjust if different
DATASET_PATH = "/content/D1_Swiss_processed.csv"
SAVE_DIR = "/content/drive/MyDrive/CAPTURE"

print("Loading saved model from:", MODEL_PATH)

checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

#recreate the autoencoder architecture
activation_map = {'ReLU': nn.ReLU, 'Tanh': nn.Tanh, 'Sigmoid': nn.Sigmoid}
activation_fn = activation_map[checkpoint['best_activation_name']]

model = Autoencoder(
    INPUT_DIM,
    checkpoint['best_hidden_size'],
    checkpoint['best_latent_dim'],
    activation_fn
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model loaded: {checkpoint['best_hidden_size']} -> {checkpoint['best_latent_dim']}")

#Load dataset
print("loading dataset...")
df = pd.read_csv(DATASET_PATH)
feature_matrix = df[FEATURE_COLUMNS].values


#split into train and test sets
train_val_data, test_data = train_test_split(
    feature_matrix,
    test_size=0.2,
    random_state=RANDOM_SEED
)

# Encode all data
print("Encoding data...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

train_val_tensor = torch.tensor(train_val_data, dtype=torch.float32)
test_tensor = torch.tensor(test_data, dtype=torch.float32)

with torch.no_grad():
    train_val_latent = model.encoder(train_val_tensor.to(device)).cpu().numpy()
    test_latent = model.encoder(test_tensor.to(device)).cpu().numpy()


print(f"Train+val latent vectors shape: {train_val_latent.shape}")
print(f"Test latent vectors shape: {test_latent.shape}")
print("clustering...")

kmeans = KMeans(n_clusters=checkpoint['best_k'], random_state=RANDOM_SEED, n_init=10)
cluster_labels_all = kmeans.fit_predict(train_val_latent)
cluster_centroids = kmeans.cluster_centers_


#Assign test samples to clusters

test_distances = cdist(test_latent, cluster_centroids, metric='euclidean')
test_cluster_assignments = np.argmin(test_distances, axis=1)

print("Computing profile summary...")
all_cluster_labels = np.concatenate([cluster_labels_all, test_cluster_assignments])
all_data = np.vstack([train_val_data, test_data])

profile_summary = []
for k in range(checkpoint['best_k']):
    cluster_mask = all_cluster_labels == k
    cluster_data = all_data[cluster_mask]
    cluster_size = np.sum(cluster_mask)
    
    # Convert to DataFrame for safe column access (FIX: use column names instead of hardcoded indices)
    cluster_df = pd.DataFrame(cluster_data, columns=FEATURE_COLUMNS)
    
    profile_summary.append({
        'Profile': f'P{k+1}',
        'N': cluster_size,
        'Depression': cluster_df['Depression'].mean(),
        'Anxiety': cluster_df['Anxiety'].mean(),
        'Stress': cluster_df['Stress'].mean(),
        'Burnout': cluster_df['Burnout'].mean()
    })

print("Profile summary computed:")
for profile in profile_summary:
    print(f"  {profile['Profile']}: N={profile['N']}, Dep={profile['Depression']:.3f}, Anx={profile['Anxiety']:.3f}, Stress={profile['Stress']:.3f}, Burnout={profile['Burnout']:.3f}")

h3_data = {
    'cluster_labels_all': cluster_labels_all,
    'test_cluster_assignments': test_cluster_assignments,
    'cluster_centroids': cluster_centroids,
    'best_k': checkpoint['best_k'],
    'best_latent_dim': checkpoint['best_latent_dim'],
    'RANDOM_SEED': RANDOM_SEED,
    'dataset_name': 'D1-Swiss',
    'profile_summary': profile_summary,
}

#h3 data saved as pickle
h3_pickle_path = os.path.join(SAVE_DIR, 'h3_data.pkl')
with open(h3_pickle_path, 'wb') as f:
    pickle.dump(h3_data, f)

print(f" H3 data saved to: {h3_pickle_path}")
print(f"  - Train+val clusters: {cluster_labels_all.shape}")
print(f"  - Test clusters: {test_cluster_assignments.shape}")
print(f"  - Centroids: {cluster_centroids.shape}")
print(f"  - Profile summary: {len(profile_summary)} profiles")


