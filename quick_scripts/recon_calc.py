import torch
import numpy as np
import pandas as pd
import joblib


DATASETS = {
    "D1-Swiss": {
        "csv": "D1_Swiss_processed_test.csv",
        "model": "D1_Swiss_model (4).pth",
        "scaler": "D1-Swiss_scaler.joblib",
        "kmeans": "D1-Swiss_kmeans_model.joblib",
        "kmeans_centroids": "D1-Swiss_kmeans_centroids.npy",
        "kmeans_meta": "D1-Swiss_kmeans_meta.pkl"
    },
    "D2-Cultural": {
        "csv": "D2_Cultural_processed_test.csv",
        "model": "D2_Cultural_model (3).pth",
        "scaler": "D2-Cultural_scaler.joblib",
        "kmeans": "D2-Cultural_kmeans_model.joblib",
        "kmeans_centroids": "D2-Cultural_kmeans_centroids.npy",
        "kmeans_meta": "D2-Cultural_kmeans_meta.pkl"
    },
    "D3-Academic": {
        "csv": "D3_Academic_processed_test.csv",
        "model": "D3_Academic_model (3).pth",
        "scaler": "D3-Academic_scaler.joblib",
        "kmeans": "D3-Academic_kmeans_model.joblib",
        "kmeans_centroids": "D3-Academic_kmeans_centroids.npy",
        "kmeans_meta": "D3-Academic_kmeans_meta.pkl"
    },
    "D4-Tech": {
        "csv": "D4_Tech_processed_test.csv",
        "model": "D4_Tech_model (3).pth ",
        "scaler": "D4-Tech_scaler.joblib",
        "kmeans": "D4-Tech_kmeans_model.joblib",
        "kmeans_centroids": "D4-Tech_kmeans_centroids.npy",
        "kmeans_meta": "D4-Tech_kmeans_meta.pkl"
    }
}

feature_columns = ["Depression", "Anxiety", "Stress", "Burnout"]

TEST_SIZE = 0.2
RANDOM_STATE = 42

activation_map = {
    'ReLU': torch.nn.ReLU,
    'Sigmoid': torch.nn.Sigmoid,
    'Tanh': torch.nn.Tanh
}

# Define Autoencoder architecture
class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, activation_function):
        super(Autoencoder, self).__init__()
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
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

print("Reconstruction Loss for Individual Models")
print("="*50)

for name, paths in DATASETS.items():
    print(f"\nEvaluating {name}...")
    
    # Load data
    df = pd.read_csv(paths["csv"])
    data = df[feature_columns].values
    
    # Load scaler and scale
    scaler = joblib.load(paths["scaler"])
    scaled = scaler.transform(data)
    
    # Load model
    checkpoint = torch.load(paths["model"], map_location='cpu', weights_only=False)
    activation_fn = activation_map[checkpoint['best_activation_name']]
    model = Autoencoder(
        4,  # INPUT_DIM
        checkpoint['best_hidden_size'],
        checkpoint['best_latent_dim'],
        activation_fn
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Compute reconstruction
    with torch.no_grad():
        tensor = torch.tensor(scaled, dtype=torch.float32)
        reconstructed = model(tensor).numpy()
    
    # MSE
    recon_loss = np.mean((scaled - reconstructed)**2)
    print(f"  Reconstruction Loss (MSE): {recon_loss:.4f}")

print("\nEvaluation Complete")
