#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ============================================================================
# CELL 0: SETUP AND CONFIGURATION
# ============================================================================
# Purpose: Initialize environment, set reproducibility seeds, define constants
# Key components:
#   - Library imports
#   - Random seed configuration for reproducibility
#   - Dataset paths and feature definitions
#   - Device selection (GPU/CPU)
#   - Data preparation function
# ============================================================================

from pathlib import Path
import json
import time
import warnings
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import chi2, chi2_contingency, pearsonr
from torch_lr_finder import LRFinder

warnings.filterwarnings('ignore')

RANDOM_SEED = 42
FEATURE_COLUMNS = ["Depression", "Anxiety", "Stress", "Burnout"]
DATASETS = {
    "D1-Swiss": Path("D1_Swiss_processed.csv"),
    "D2-Cultural": Path("D2_Cultural_processed.csv"),
    "D3-Academic": Path("D3_Academic_processed.csv"),
    "D4-Tech": Path("D4_Tech_processed.csv"),
}
# Optimal bin numbers for stratified splitting (from grid search)
STRATIFICATION_BINS = {
    "D1-Swiss": 2,
    "D2-Cultural": 2,
    "D3-Academic": 4,
    "D4-Tech": 2,
}
PIPELINE_RESULTS = {}

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

def prepare_dataset(dataset_name: str):
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(DATASETS.keys())}")

    dataset_path = DATASETS[dataset_name]
    print(f"\n=== Loading {dataset_name} dataset ===")
    print(f"File: {dataset_path}")
    print(f"Features: {FEATURE_COLUMNS}")

    df = pd.read_csv(dataset_path)

    # Check if required columns exist
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {dataset_path}: {missing_cols}")

    # Check if dataset is empty
    if len(df) == 0:
        raise ValueError(f"Dataset {dataset_path} is empty.")

    feature_matrix = df[FEATURE_COLUMNS].values
    print(f"Dataset: {feature_matrix.shape[0]} samples, {feature_matrix.shape[1]} features")

    # Check if we have enough samples for train/test split
    if len(feature_matrix) < 10:
        raise ValueError(f"Dataset too small ({len(feature_matrix)} samples). Need at least 10 samples.")

    # ========================================================================
    # STRATIFIED SPLIT: Use binned features for stratification
    # ========================================================================
    # Create bins for each continuous feature to enable stratification
    # This ensures train/test sets have similar distributions
    # ========================================================================
    try:
        n_bins = STRATIFICATION_BINS.get(dataset_name, 2)  # Default to 2 bins if not specified
        
        # Create bins for each feature (using quantiles)
        df_binned = df.copy()
        for col in FEATURE_COLUMNS:
            # Use quantiles to create bins, handle duplicates
            df_binned[f'{col}_bin'] = pd.qcut(df[col], q=n_bins, labels=False, duplicates='drop')
        
        # Create stratification label (combination of all feature bins)
        df_binned['stratify_label'] = df_binned[[f'{col}_bin' for col in FEATURE_COLUMNS]].apply(
            lambda x: '_'.join(x.astype(str)), axis=1
        )
        
        # Check if we have enough samples per stratum for stratification
        stratum_counts = df_binned['stratify_label'].value_counts()
        min_stratum_size = stratum_counts.min()
        
        if min_stratum_size < 2:
            raise ValueError(f"Some strata have < 2 samples (min = {min_stratum_size}), cannot stratify")
        
        # Perform stratified split
        train_val_data, test_data = train_test_split(
            feature_matrix, 
            test_size=0.2, 
            random_state=RANDOM_SEED,
            stratify=df_binned['stratify_label']
        )
        print(f"✓ Using STRATIFIED split with {n_bins} bins per feature (ensures balanced train/test distributions)")
        
    except (ValueError, KeyError) as e:
        # Fallback to regular split if stratification fails
        print(f"⚠ Stratification failed ({str(e)}), using regular split")
        train_val_data, test_data = train_test_split(
            feature_matrix, 
            test_size=0.2, 
            random_state=RANDOM_SEED
        )

    train_val_tensor = torch.tensor(train_val_data, dtype=torch.float32)
    test_tensor = torch.tensor(test_data, dtype=torch.float32)
    kfold = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

    print(f"Train+Val: {train_val_tensor.shape[0]} samples (80%)")
    print(f"Test: {test_tensor.shape[0]} samples (20%)")
    print(f"K-Fold: 10 folds, ~{train_val_tensor.shape[0]//10} samples per fold\n")

    return (
        df,
        feature_matrix,
        train_val_data,
        test_data,
        train_val_tensor,
        test_tensor,
        kfold,
        dataset_path,
    )

INPUT_DIM = len(FEATURE_COLUMNS)


# In[ ]:


# ============================================================================
# CELL 1: AUTOENCODER ARCHITECTURE DEFINITION
# ============================================================================
# Purpose: Define the neural network that compresses 4D symptoms into
#          lower-dimensional latent space for clustering
# Architecture: Symmetric encoder-decoder with configurable dimensions
# ============================================================================

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, activation_function):
        super(Autoencoder, self).__init__()

        #Encoder - input -> hidden -> latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation_function(),
            nn.Linear(hidden_dim, latent_dim)
        )


        #Decoder - latent -> hidden -> output
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            activation_function(),
            nn.Linear(hidden_dim, input_dim)
        )


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# In[ ]:


# ============================================================================
# CELL 2: MAIN PIPELINE - Hyperparameter Tuning and Profile Extraction
# ============================================================================
# Purpose: Two-stage hyperparameter optimization followed by profile extraction
# Stage 1: Architecture tuning (hidden_size, latent_dim, activation, optimizer)
# Stage 2: Learning parameter tuning (batch_size, weight_decay, momentum)
# Final: Train on all data, extract profiles, evaluate on test set
# ============================================================================

def run_autoencoder_pipeline(dataset_name: str, force_latent_dim: int = None, force_k: int = None):
    (
        all_data_df,
        all_data,
        train_val_data,
        test_data,
        train_val_tensor,
        test_tensor,
        kfold,
        dataset_path,
    ) = prepare_dataset(dataset_name)

    # ------------------------------------------------------------------------
    # STAGE 1: Architecture Parameter Tuning
    # ------------------------------------------------------------------------
    # Grid search over architecture parameters with 10-fold cross-validation
    # Selection criterion: Consensus voting on clustering quality metrics
    # ------------------------------------------------------------------------
    print("STAGE 1: Architecture Parameters Tuning (K-Fold CV)")
    print("="*70)

    hidden_sizes = [3, 4, 5, 6, 8, 10]
    latent_dims = [force_latent_dim] if force_latent_dim is not None else [2, 3]

    activations = {
        'ReLU': nn.ReLU,
        'Tanh': nn.Tanh,
        'Sigmoid': nn.Sigmoid
    }

    optimizers = {
        'Adam': optim.Adam,
        'SGD': optim.SGD
    }

    epochs_list = [50, 75, 100]
    fixed_lr = 1e-3

    n_folds = 10
    criterion = nn.MSELoss()
    results_stage1 = defaultdict(list)

    total_experiments = len(hidden_sizes) * len(latent_dims) * len(activations) * len(optimizers) * len(epochs_list) * n_folds
    experiment_count = 0

    print(f"Testing: hidden_size, latent_dim, activation, optimizer, epochs")
    print(f"Fixed: learning_rate = {fixed_lr}")
    print(f"\nGrid sizes:")
    print(f"  hidden_size: {len(hidden_sizes)} values {hidden_sizes}")
    print(f"  latent_dim: {len(latent_dims)} values {latent_dims}")
    print(f"  activation: {len(activations)} values {list(activations.keys())}")
    print(f"  optimizer: {len(optimizers)} values {list(optimizers.keys())}")
    print(f"  epochs: {len(epochs_list)} values {epochs_list}")
    print(f"  CV folds: {n_folds}")
    print(f"Total: {total_experiments} experiments ({total_experiments//n_folds} configs × {n_folds} folds)\n")

    start_stage1 = time.time()

    for hidden_size in hidden_sizes:
        for latent_dim in latent_dims:
            for act_name, act_fn in activations.items():
                for opt_name, opt_class in optimizers.items():
                    for num_epochs in epochs_list:
                        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(train_val_tensor)):
                            experiment_count += 1
                            if experiment_count % 50 == 0 or experiment_count == total_experiments:
                                elapsed = time.time() - start_stage1
                                print(f"  [{experiment_count}/{total_experiments}] {100*experiment_count/total_experiments:.1f}% - {elapsed/60:.1f}min")

                            fold_seed = 42 + fold_idx
                            torch.manual_seed(fold_seed)
                            np.random.seed(fold_seed)

                            train_fold = train_val_tensor[train_idx]
                            val_fold = train_val_tensor[val_idx]

                            train_dataset = TensorDataset(train_fold.cpu())
                            val_dataset = TensorDataset(val_fold.cpu())
                            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
                            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

                            model = Autoencoder(INPUT_DIM, hidden_size, latent_dim, act_fn).to(device)

                            if opt_name == 'Adam':
                                optimizer = opt_class(model.parameters(), lr=fixed_lr)
                            else:
                                optimizer = opt_class(model.parameters(), lr=fixed_lr, momentum=0.9)

                            train_losses, val_losses, optimal_k, best_sil_score, latent_vectors, validation_metrics = \
                                train_and_validate_model(model, train_loader, val_loader, optimizer, 
                                                        criterion, num_epochs, device)

                            # Safely get index for optimal_k
                            if optimal_k in validation_metrics['k_values']:
                                optimal_k_idx = validation_metrics['k_values'].index(optimal_k)
                                best_ch_score = validation_metrics['calinski_harabasz_scores'][optimal_k_idx]
                                best_db_score = validation_metrics['davies_bouldin_scores'][optimal_k_idx]
                            else:
                                # Fallback: use first available K
                                optimal_k_idx = 0
                                best_ch_score = validation_metrics['calinski_harabasz_scores'][0] if len(validation_metrics['calinski_harabasz_scores']) > 0 else 0
                                best_db_score = validation_metrics['davies_bouldin_scores'][0] if len(validation_metrics['davies_bouldin_scores']) > 0 else float('inf')

                            results_stage1['hidden_size'].append(hidden_size)
                            results_stage1['latent_dim'].append(latent_dim)
                            results_stage1['activation'].append(act_name)
                            results_stage1['optimizer'].append(opt_name)
                            results_stage1['epochs'].append(num_epochs)
                            results_stage1['fold'].append(fold_idx)
                            results_stage1['optimal_k'].append(optimal_k)
                            results_stage1['silhouette_score'].append(best_sil_score)
                            results_stage1['calinski_harabasz_score'].append(best_ch_score)
                            results_stage1['davies_bouldin_score'].append(best_db_score)
                            results_stage1['reconstruction_loss'].append(val_losses[-1])
                            results_stage1['consensus_reached'].append(validation_metrics['consensus_reached'])

    print(f"\n✓ Stage 1 completed in {(time.time()-start_stage1)/60:.2f} minutes\n")

    # ------------------------------------------------------------------------
    # STAGE 1: Results Aggregation and Best Configuration Selection
    # ------------------------------------------------------------------------
    # Aggregate results across folds, rank by clustering metrics,
    # use consensus voting to select best architecture
    # ------------------------------------------------------------------------
    print("STAGE 1: Results Aggregation")
    print("="*70)

    stage1_df = pd.DataFrame(results_stage1)

    # Check if we have any results
    if len(stage1_df) == 0:
        raise ValueError("Stage 1 produced no results. Check training loop.")

    aggregated_stage1 = stage1_df.groupby(['hidden_size', 'latent_dim', 'activation', 'optimizer', 'epochs']).agg({
        'silhouette_score': ['mean', 'std'],
        'calinski_harabasz_score': ['mean', 'std'],
        'davies_bouldin_score': ['mean', 'std'],
        'reconstruction_loss': ['mean', 'std'],
        'optimal_k': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
    }).reset_index()

    aggregated_stage1.columns = ['hidden_size', 'latent_dim', 'activation', 'optimizer', 'epochs',
                                  'mean_silhouette', 'std_silhouette',
                                  'mean_ch', 'std_ch',
                                  'mean_db', 'std_db',
                                  'mean_recon_loss', 'std_recon_loss',
                                  'most_common_k']

    # Create config_id BEFORE filtering
    aggregated_stage1['config_id'] = aggregated_stage1.apply(
        lambda row: f"{row['hidden_size']}_{row['latent_dim']}_{row['activation']}_{row['optimizer']}_{row['epochs']}", 
        axis=1
    )

    aggregated_stage1['rank_silhouette'] = aggregated_stage1['mean_silhouette'].rank(ascending=False, method='min')
    aggregated_stage1['rank_ch'] = aggregated_stage1['mean_ch'].rank(ascending=False, method='min')
    aggregated_stage1['rank_db'] = aggregated_stage1['mean_db'].rank(ascending=True, method='min')

    top_by_sil = aggregated_stage1.loc[aggregated_stage1['rank_silhouette'] == 1]
    top_by_ch = aggregated_stage1.loc[aggregated_stage1['rank_ch'] == 1]
    top_by_db = aggregated_stage1.loc[aggregated_stage1['rank_db'] == 1]

    votes = []
    if len(top_by_sil) > 0:
        votes.extend(top_by_sil['config_id'].tolist())
    if len(top_by_ch) > 0:
        votes.extend(top_by_ch['config_id'].tolist())
    if len(top_by_db) > 0:
        votes.extend(top_by_db['config_id'].tolist())

    # Check if aggregated_stage1 is empty
    if len(aggregated_stage1) == 0:
        raise ValueError("Stage 1 aggregation produced no results. Check groupby operation.")

    vote_counts = Counter(votes)
    if len(vote_counts) > 0:
        most_voted_config, vote_count = vote_counts.most_common(1)[0]

        if vote_count >= 2:
            matching_configs = aggregated_stage1[aggregated_stage1['config_id'] == most_voted_config]
            if len(matching_configs) == 0:
                raise ValueError(f"Config {most_voted_config} not found in aggregated results.")
            best_config_stage1 = matching_configs.iloc[0]
            consensus_status = f"Consensus: {vote_count} metrics agree"
        else:
            top_sil = aggregated_stage1.loc[aggregated_stage1['rank_silhouette'] == 1]
            if len(top_sil) == 0:
                raise ValueError("No top silhouette config found.")
            best_config_stage1 = top_sil.iloc[0]
            consensus_status = f"No consensus. Using Silhouette"
    else:
        aggregated_stage1_sorted = aggregated_stage1.sort_values('mean_silhouette', ascending=False)
        if len(aggregated_stage1_sorted) == 0:
            raise ValueError("No configurations to select from.")
        best_config_stage1 = aggregated_stage1_sorted.iloc[0]
        consensus_status = "Using Silhouette (fallback)"

    best_hidden_size = int(best_config_stage1['hidden_size'])
    best_latent_dim = int(best_config_stage1['latent_dim'])
    best_activation_name = best_config_stage1['activation']
    best_optimizer_name = best_config_stage1['optimizer']
    best_epochs = int(best_config_stage1['epochs'])
    best_k = force_k if force_k is not None else int(best_config_stage1['most_common_k'])

    print(f"Top 5 configs (by mean silhouette score):\n")
    aggregated_stage1_sorted = aggregated_stage1.sort_values('mean_silhouette', ascending=False)
    print(aggregated_stage1_sorted.head(5)[['hidden_size', 'latent_dim', 'activation', 'optimizer', 
                                            'epochs', 'mean_silhouette', 'std_silhouette',
                                            'mean_ch', 'mean_db', 'most_common_k']].to_string(index=False))

    print(f"\n{'='*70}")
    print(f"Best Architecture (Stage 1): {consensus_status}")
    print(f"  hidden_size={best_hidden_size}, latent_dim={best_latent_dim}")
    print(f"  activation={best_activation_name}, optimizer={best_optimizer_name}, epochs={best_epochs}")
    print(f"  Silhouette: {best_config_stage1['mean_silhouette']:.6f} ± {best_config_stage1['std_silhouette']:.6f}")
    print(f"  Calinski-Harabasz: {best_config_stage1['mean_ch']:.6f} ± {best_config_stage1['std_ch']:.6f}")
    print(f"  Davies-Bouldin: {best_config_stage1['mean_db']:.6f} ± {best_config_stage1['std_db']:.6f}")
    print(f"  Reconstruction loss: {best_config_stage1['mean_recon_loss']:.6f} ± {best_config_stage1['std_recon_loss']:.6f}")
    print(f"  Most common optimal K: {best_k}")
    print(f"{'='*70}\n")

    # ------------------------------------------------------------------------
    # STAGE 2: Learning Parameter Optimization
    # ------------------------------------------------------------------------
    # Fine-tune training parameters using best architecture from Stage 1
    # Learning rate determined via LR Range Test
    # ------------------------------------------------------------------------
    print("STAGE 2: Learning Parameter Optimization (K-Fold CV)")
    print("="*70)
    print(f"Using best architecture from Stage 1:")
    print(f"  hidden={best_hidden_size}, latent={best_latent_dim}, activation={best_activation_name}, optimizer={best_optimizer_name}")

    activation_map = {'ReLU': nn.ReLU, 'Tanh': nn.Tanh, 'Sigmoid': nn.Sigmoid}
    best_activation_fn = activation_map[best_activation_name]

    # ------------------------------------------------------------------------
    # LR Range Test: Find Optimal Learning Rate
    # ------------------------------------------------------------------------
    # Test learning rates from 1e-7 to 10 to find optimal value
    # Uses subset of training data for efficiency
    # ------------------------------------------------------------------------
    print("\n" + "="*70)
    print("LR RANGE TEST (Using Best Architecture from Stage 1)")
    print("="*70)

    # Create DataLoader that returns (input, target) pairs for LR finder
    # For autoencoder, target = input (reconstruction task)
    train_subset_data = train_val_tensor[:500]
    train_subset = TensorDataset(train_subset_data, train_subset_data)  # (input, target) where target=input
    train_loader_lr = DataLoader(train_subset, batch_size=64, shuffle=True)
    criterion = nn.MSELoss()

    print(f"\nLR Range Test: {best_optimizer_name} optimizer")
    model_lr = Autoencoder(INPUT_DIM, best_hidden_size, best_latent_dim, best_activation_fn).to(device)

    if best_optimizer_name == 'Adam':
        optimizer_lr = optim.Adam(model_lr.parameters(), lr=1e-7)
    else:
        optimizer_lr = optim.SGD(model_lr.parameters(), lr=1e-7, momentum=0.9)

    lr_finder = LRFinder(model_lr, optimizer_lr, criterion, device=device)
    lr_finder.range_test(train_loader_lr, end_lr=10, num_iter=100, step_mode="exp")
    lr_finder.plot()
    plt.title(f'{best_optimizer_name} LR Range Test (Best Architecture)', fontsize=14, fontweight='bold')
    plt.show()

    history = lr_finder.history
    lrs = np.array(history['lr'])
    losses = np.array(history['loss'])
    loss_diffs = np.diff(losses)
    descending = np.where(loss_diffs < 0)[0]

    if len(descending) > 0:
        mid = (descending[0] + descending[-1]) // 2
        optimal_lr = lrs[mid]
    else:
        optimal_lr = lrs[np.argmin(losses)]

    print(f"Optimal LR from range test: {optimal_lr:.2e}\n")

    lr_finder.reset()

    batch_sizes = [32, 64, 128]
    weight_decays = [0, 1e-4, 1e-3]

    if best_optimizer_name == 'SGD':
        momentum_values = [0.5, 0.9, 0.95]
        print(f"Testing: batch_size, weight_decay, momentum")
        print(f"Fixed: learning_rate = {optimal_lr:.2e}")
        total_experiments = len(batch_sizes) * len(weight_decays) * len(momentum_values) * n_folds
    else:
        momentum_values = [None]
        print(f"Testing: batch_size, weight_decay")
        print(f"Fixed: learning_rate = {optimal_lr:.2e}")
        total_experiments = len(batch_sizes) * len(weight_decays) * n_folds

    results_stage2 = defaultdict(list)
    experiment_count = 0

    print(f"\nGrid sizes:")
    print(f"  learning_rate: 1 value (optimal from LR range test)")
    print(f"  batch_size: {len(batch_sizes)} values {batch_sizes}")
    print(f"  weight_decay: {len(weight_decays)} values {weight_decays}")
    if best_optimizer_name == 'SGD':
        print(f"  momentum: {len(momentum_values)} values {momentum_values}")
    print(f"  CV folds: {n_folds}")
    print(f"Total: {total_experiments} experiments ({total_experiments//n_folds} configs × {n_folds} folds)\n")

    start_stage2 = time.time()

    for batch_size in batch_sizes:
        for weight_decay in weight_decays:
            for momentum in momentum_values:
                for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(train_val_tensor)):
                    experiment_count += 1
                    if experiment_count % 50 == 0 or experiment_count == total_experiments:
                        elapsed = time.time() - start_stage2
                        print(f"  [{experiment_count}/{total_experiments}] {100*experiment_count/total_experiments:.1f}% - {elapsed/60:.1f}min")

                    fold_seed = 42 + fold_idx
                    torch.manual_seed(fold_seed)
                    np.random.seed(fold_seed)

                    train_fold = train_val_tensor[train_idx]
                    val_fold = train_val_tensor[val_idx]

                    train_dataset = TensorDataset(train_fold.cpu())
                    val_dataset = TensorDataset(val_fold.cpu())
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                    model = Autoencoder(INPUT_DIM, best_hidden_size, best_latent_dim, best_activation_fn).to(device)

                    if best_optimizer_name == 'SGD':
                        optimizer = optim.SGD(model.parameters(), lr=optimal_lr, momentum=momentum, weight_decay=weight_decay)
                    else:
                        optimizer = optim.Adam(model.parameters(), lr=optimal_lr, weight_decay=weight_decay)

                    train_losses, val_losses, optimal_k, best_sil_score, latent_vectors, validation_metrics = \
                        train_and_validate_model(model, train_loader, val_loader, optimizer, 
                                                criterion, best_epochs, device)

                    # Safely get index for optimal_k
                    if optimal_k in validation_metrics['k_values']:
                        optimal_k_idx = validation_metrics['k_values'].index(optimal_k)
                        best_ch_score = validation_metrics['calinski_harabasz_scores'][optimal_k_idx]
                        best_db_score = validation_metrics['davies_bouldin_scores'][optimal_k_idx]
                    else:
                        # Fallback: use first available K
                        optimal_k_idx = 0
                        best_ch_score = validation_metrics['calinski_harabasz_scores'][0] if len(validation_metrics['calinski_harabasz_scores']) > 0 else 0
                        best_db_score = validation_metrics['davies_bouldin_scores'][0] if len(validation_metrics['davies_bouldin_scores']) > 0 else float('inf')

                    results_stage2['learning_rate'].append(optimal_lr)
                    results_stage2['batch_size'].append(batch_size)
                    results_stage2['weight_decay'].append(weight_decay)
                    if best_optimizer_name == 'SGD':
                        results_stage2['momentum'].append(momentum)
                    results_stage2['fold'].append(fold_idx)
                    results_stage2['silhouette_score'].append(best_sil_score)
                    results_stage2['calinski_harabasz_score'].append(best_ch_score)
                    results_stage2['davies_bouldin_score'].append(best_db_score)
                    results_stage2['reconstruction_loss'].append(val_losses[-1])

    print(f"\n✓ Stage 2 completed in {(time.time()-start_stage2)/60:.2f} minutes\n")

    # ------------------------------------------------------------------------
    # STAGE 2: Results Aggregation and Best Configuration Selection
    # ------------------------------------------------------------------------
    # Aggregate results across folds, select best learning parameters
    # ------------------------------------------------------------------------
    print("STAGE 2: Results Aggregation")
    print("="*70)

    stage2_df = pd.DataFrame(results_stage2)

    # Check if we have any results
    if len(stage2_df) == 0:
        raise ValueError("Stage 2 produced no results. Check training loop.")

    if 'momentum' in results_stage2:
        groupby_cols = ['learning_rate', 'batch_size', 'weight_decay', 'momentum']
    else:
        groupby_cols = ['learning_rate', 'batch_size', 'weight_decay']

    aggregated_stage2 = stage2_df.groupby(groupby_cols).agg({
        'silhouette_score': ['mean', 'std'],
        'calinski_harabasz_score': ['mean', 'std'],
        'davies_bouldin_score': ['mean', 'std'],
        'reconstruction_loss': ['mean', 'std']
    }).reset_index()

    col_names = groupby_cols + ['mean_silhouette', 'std_silhouette', 'mean_ch', 'std_ch', 
                                'mean_db', 'std_db', 'mean_recon_loss', 'std_recon_loss']
    aggregated_stage2.columns = col_names

    # Create config_id BEFORE filtering
    # Check if aggregated_stage2 is empty
    if len(aggregated_stage2) == 0:
        raise ValueError("Stage 2 aggregation produced no results. Check groupby operation.")

    aggregated_stage2['config_id'] = aggregated_stage2.apply(
        lambda row: f"{row['learning_rate']:.2e}_{row['batch_size']}_{row['weight_decay']:.2e}" + 
                    (f"_{row['momentum']}" if 'momentum' in row.index else ""), 
        axis=1
    )

    aggregated_stage2['rank_silhouette'] = aggregated_stage2['mean_silhouette'].rank(ascending=False, method='min')
    aggregated_stage2['rank_ch'] = aggregated_stage2['mean_ch'].rank(ascending=False, method='min')
    aggregated_stage2['rank_db'] = aggregated_stage2['mean_db'].rank(ascending=True, method='min')

    top_by_sil = aggregated_stage2.loc[aggregated_stage2['rank_silhouette'] == 1]
    top_by_ch = aggregated_stage2.loc[aggregated_stage2['rank_ch'] == 1]
    top_by_db = aggregated_stage2.loc[aggregated_stage2['rank_db'] == 1]

    votes = []
    if len(top_by_sil) > 0:
        votes.extend(top_by_sil['config_id'].tolist())
    if len(top_by_ch) > 0:
        votes.extend(top_by_ch['config_id'].tolist())
    if len(top_by_db) > 0:
        votes.extend(top_by_db['config_id'].tolist())

    vote_counts = Counter(votes)
    if len(vote_counts) > 0:
        most_voted_config, vote_count = vote_counts.most_common(1)[0]

        if vote_count >= 2:
            matching_configs = aggregated_stage2[aggregated_stage2['config_id'] == most_voted_config]
            if len(matching_configs) == 0:
                raise ValueError(f"Config {most_voted_config} not found in aggregated results.")
            best_config_stage2 = matching_configs.iloc[0]
            consensus_status = f"Consensus: {vote_count} metrics agree"
        else:
            top_sil = aggregated_stage2.loc[aggregated_stage2['rank_silhouette'] == 1]
            if len(top_sil) == 0:
                raise ValueError("No top silhouette config found.")
            best_config_stage2 = top_sil.iloc[0]
            consensus_status = f"No consensus. Using Silhouette"
    else:
        aggregated_stage2_sorted = aggregated_stage2.sort_values('mean_silhouette', ascending=False)
        if len(aggregated_stage2_sorted) == 0:
            raise ValueError("No configurations to select from.")
        best_config_stage2 = aggregated_stage2_sorted.iloc[0]
        consensus_status = "Using Silhouette (fallback)"

    best_learning_rate = best_config_stage2['learning_rate']
    best_batch_size = int(best_config_stage2['batch_size'])
    best_weight_decay = best_config_stage2['weight_decay']
    best_momentum = best_config_stage2.get('momentum', None)

    print(f"Top 5 configs (by mean silhouette score):\n")
    aggregated_stage2_sorted = aggregated_stage2.sort_values('mean_silhouette', ascending=False)
    print(aggregated_stage2_sorted.head(5)[groupby_cols + ['mean_silhouette', 'std_silhouette', 
                                                            'mean_ch', 'mean_db']].to_string(index=False))

    print(f"\n{'='*70}")
    print(f"Best Overall Configuration: {consensus_status}")
    print(f"{'='*70}")
    print(f"Architecture (from Stage 1):")
    print(f"  hidden_size={best_hidden_size}, latent_dim={best_latent_dim}")
    print(f"  activation={best_activation_name}, optimizer={best_optimizer_name}, epochs={best_epochs}")
    print(f"\nLearning Parameters (from Stage 2):")
    print(f"  learning_rate={best_learning_rate:.2e}")
    print(f"  batch_size={best_batch_size}")
    print(f"  weight_decay={best_weight_decay:.2e}", end="")
    if best_momentum is not None:
        print(f", momentum={best_momentum}")
    else:
        print()
    print(f"\nPerformance:")
    print(f"  Silhouette: {best_config_stage2['mean_silhouette']:.6f} ± {best_config_stage2['std_silhouette']:.6f}")
    print(f"  Calinski-Harabasz: {best_config_stage2['mean_ch']:.6f} ± {best_config_stage2['std_ch']:.6f}")
    print(f"  Davies-Bouldin: {best_config_stage2['mean_db']:.6f} ± {best_config_stage2['std_db']:.6f}")
    print(f"  Reconstruction loss: {best_config_stage2['mean_recon_loss']:.6f} ± {best_config_stage2['std_recon_loss']:.6f}")
    print(f"  Optimal K: {best_k}")
    print(f"{'='*70}\n")

    # ------------------------------------------------------------------------
    # FINAL MODEL TRAINING AND LATENT PROFILE EXTRACTION
    # ------------------------------------------------------------------------
    # Train final model on all train+val data with best hyperparameters
    # Extract latent vectors and perform K-means clustering
    # ------------------------------------------------------------------------
    print("Final Model Training and Latent Profile Extraction")
    print("="*70)
    print(f"Training final model on all {dataset_name} train+val data with best hyperparameters:")
    print(f"  Architecture: hidden={best_hidden_size}, latent={best_latent_dim}, activation={best_activation_name}")
    print(f"  Optimizer: {best_optimizer_name}, lr={best_learning_rate:.2e}, batch_size={best_batch_size}")
    print(f"  Epochs: {best_epochs}, Optimal K: {best_k}")
    print("="*70)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    final_dataset = TensorDataset(train_val_tensor.cpu())
    final_loader = DataLoader(final_dataset, batch_size=best_batch_size, shuffle=True)

    activation_map = {'ReLU': nn.ReLU, 'Tanh': nn.Tanh, 'Sigmoid': nn.Sigmoid}
    best_activation_fn = activation_map[best_activation_name]
    final_model = Autoencoder(INPUT_DIM, best_hidden_size, best_latent_dim, best_activation_fn).to(device)

    if best_optimizer_name == 'SGD':
        final_optimizer = optim.SGD(final_model.parameters(), lr=best_learning_rate, 
                                    momentum=best_momentum if best_momentum is not None else 0.9, 
                                    weight_decay=best_weight_decay)
    else:
        final_optimizer = optim.Adam(final_model.parameters(), lr=best_learning_rate, 
                                    weight_decay=best_weight_decay)

    criterion = nn.MSELoss()

    print(f"\nTraining final model...")
    final_model.train()
    final_losses = []

    for epoch in range(best_epochs):
        epoch_loss = 0.0
        for batch_data in final_loader:
            batch_data = batch_data[0].to(device)
            final_optimizer.zero_grad()
            reconstructed = final_model(batch_data)
            loss = criterion(reconstructed, batch_data)
            loss.backward()
            final_optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(final_loader)
        final_losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{best_epochs}, Loss: {avg_loss:.4f}")

    print(f"\nFinal model training complete. Saving model...")
    final_model.eval()
    all_latent_vectors_batches = []

    with torch.no_grad():
        for batch_data in final_loader:
            data = batch_data[0].to(device)
            latent = final_model.encoder(data)
            all_latent_vectors_batches.append(latent.cpu())
    latent_vectors_all = np.vstack(all_latent_vectors_batches)
    print(f"Latent vectors shape: {latent_vectors_all.shape}")

    # ------------------------------------------------------------------------
    # K-Means Clustering: Extract Mental Health Profiles
    # ------------------------------------------------------------------------
    # Cluster latent vectors to identify distinct mental health profiles
    # ------------------------------------------------------------------------
    print(f"Running K-means clustering with optimal k... {best_k}")
    final_kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    cluster_labels_all = final_kmeans.fit_predict(latent_vectors_all)
    cluster_centroids = final_kmeans.cluster_centers_

    print(f"Cluster Assignments: {cluster_labels_all}")
    print(f"Cluster Centroids: {cluster_centroids.shape}")

    # Check if we have at least 2 clusters (required for silhouette score)
    n_unique_clusters = len(np.unique(cluster_labels_all))
    if n_unique_clusters < 2:
        print(f"⚠️ ERROR: Final clustering produced only {n_unique_clusters} cluster(s). Cannot compute metrics.")
        final_sil_score = -1
        final_ch_score = 0
        final_db_score = float('inf')
    else:
        final_sil_score = silhouette_score(latent_vectors_all, cluster_labels_all)
        print(f"Final Silhouette Score: {final_sil_score:.4f}")
        final_ch_score = calinski_harabasz_score(latent_vectors_all, cluster_labels_all)
        print(f"Final Calinski-Harabasz Score: {final_ch_score:.4f}")
        final_db_score = davies_bouldin_score(latent_vectors_all, cluster_labels_all)
        print(f"Final Davies-Bouldin Score: {final_db_score:.4f}")

    # ------------------------------------------------------------------------
    # PROFILE CHARACTERISTICS EXTRACTION
    # ------------------------------------------------------------------------
    # Map cluster labels back to original symptom space to interpret profiles
    # Compute mean symptom levels (Depression, Anxiety, Stress, Burnout) per cluster
    # ------------------------------------------------------------------------
    print(f"\nProfile Characteristics (mean feature values per cluster):")
    print("="*70)

    # VERIFICATION: Check that column order matches FEATURE_COLUMNS
    print("Verifying feature column order...")
    sample_0 = train_val_data[0]
    print(f"FEATURE_COLUMNS: {FEATURE_COLUMNS}")
    print(f"First sample values:")
    for i, feature_name in enumerate(FEATURE_COLUMNS):
        print(f"  train_val_data[0, {i}] = {sample_0[i]:.4f} → {feature_name}")
    print("="*70 + "\n")

    profile_summary = []

    for k in range(best_k):
        cluster_mask = cluster_labels_all == k
        # CRITICAL: Use original 4D symptom data, NOT latent vectors as they are more intellgible to understand  
        cluster_data = train_val_data[cluster_mask]  # Original 4D: [Depression, Anxiety, Stress, Burnout]
        cluster_size = np.sum(cluster_mask)

        # Method 1: Using array indexing (for verification)
        feature_means_array = cluster_data.mean(axis=0)

        # Method 2: Using DataFrame for explicit mapping (SAFER)
        cluster_df = pd.DataFrame(cluster_data, columns=FEATURE_COLUMNS)
        feature_means_dict = cluster_df.mean().to_dict()

        # Verify they match
        print(f"Cluster {k} (N={cluster_size}):")
        for i, feature_name in enumerate(FEATURE_COLUMNS):
            array_val = feature_means_array[i]
            dict_val = feature_means_dict[feature_name]
            match = " MATCH" if abs(array_val - dict_val) < 1e-10 else " MISMATCH"
            print(f"  Index {i} ({feature_name}): array[{i}]={array_val:.6f}, dict['{feature_name}']={dict_val:.6f} {match}")

        # Use dictionary approach (explicit, no index guessing)
        profile_summary.append({
            'Profile': f'P{k+1}',
            'N': cluster_size,
            'Depression': feature_means_dict['Depression'],
            'Anxiety': feature_means_dict['Anxiety'],
            'Stress': feature_means_dict['Stress'],
            'Burnout': feature_means_dict['Burnout']
        })
        print()

    profile_df = pd.DataFrame(profile_summary)
    print("Profile Summary Table:")
    print(profile_df.to_string(index=False))

    print(f"\n{'='*70}")
    print("Final Model Results Saved:")
    print(f"  - {dataset_name} latent vectors: {latent_vectors_all.shape}")
    print(f"  - {dataset_name} cluster assignments: {cluster_labels_all.shape}")
    print(f"  - {dataset_name} cluster centroids: {cluster_centroids.shape}")
    print(f"  - Profile summary: {len(profile_summary)} profiles")
    print(f"{'='*70}\n")

    # ------------------------------------------------------------------------
    # PROFILE INTERPRETATION
    # ------------------------------------------------------------------------
    # Classify profiles based on symptom levels relative to global thresholds
    # Assign meaningful names (e.g., "Severe Comorbid", "Low Symptom")
    # ------------------------------------------------------------------------
    print("Interpretation of the profiles:")
    print("="*70)


    #It is important to compute global threshold values for each profile based on the entire dataset not just the cluster data
    #This ensures consistency and comparability across different datasets


    global_depression_threshold_high = np.percentile(train_val_data[:, 0], 75)  
    global_depression_threshold_low = np.percentile(train_val_data[:, 0], 25)  
    global_anxiety_threshold_high = np.percentile(train_val_data[:, 1], 75)
    global_anxiety_threshold_low = np.percentile(train_val_data[:, 1], 25)
    global_stress_threshold_high = np.percentile(train_val_data[:, 2], 75)
    global_stress_threshold_low = np.percentile(train_val_data[:, 2], 25)
    global_burnout_threshold_high = np.percentile(train_val_data[:, 3], 75)
    global_burnout_threshold_low = np.percentile(train_val_data[:, 3], 25)

    def interpret_profile(depression, anxiety, stress, burnout):
        """
        Interpret a single profile based on global thresholds
        returns a string description of the profile
        """

        high_symptoms = []
        low_symptoms = []


        # Compare to global thresholds not just cluster centroids
        if depression > global_depression_threshold_high:
            high_symptoms.append("Depression")
        elif depression < global_depression_threshold_low:
            low_symptoms.append("Depression")

        if anxiety > global_anxiety_threshold_high:
            high_symptoms.append("Anxiety")
        elif anxiety < global_anxiety_threshold_low:
            low_symptoms.append("Anxiety")

        if stress > global_stress_threshold_high:
            high_symptoms.append("Stress")
        elif stress < global_stress_threshold_low:
            low_symptoms.append("Stress")

        if burnout > global_burnout_threshold_high:
            high_symptoms.append("Burnout")
        elif burnout < global_burnout_threshold_low:
            low_symptoms.append("Burnout")

        if len(high_symptoms) >= 3:
            return "Severe Comorbid Profile", "High levels across multiple dimensions"
        elif "Depression" in high_symptoms and "Anxiety" in high_symptoms:
            return "Depression-Anxiety Comorbidity Profile", "High Depression and Anxiety, typical of internalizing disorders"
        elif "Stress" in high_symptoms and "Burnout" in high_symptoms:
            return "Stress-Burnout Profile", "High Stress and Burnout, typical of work-related distress"
        elif len(high_symptoms) == 1:
            return f"High {high_symptoms[0]} Profile", f"Elevated {high_symptoms[0]} with other symptoms in normal range"
        elif len(low_symptoms) >= 3:
            return "Low Symptom Profile", "Low levels across most dimensions"
        else:
            return "Moderate/Mixed Profile", "Moderate levels across dimensions"

    for i, profile in enumerate(profile_summary):
        profile_name, description = interpret_profile(
            profile['Depression'], 
            profile['Anxiety'], 
            profile['Stress'], 
            profile['Burnout']
        )
        profile_summary[i]['Profile_Name'] = profile_name
        profile_summary[i]['Description'] = description

    # Display interpreted profiles
    print("Interpreted Profiles:")
    interpreted_df = pd.DataFrame(profile_summary)
    print(interpreted_df[['Profile', 'Profile_Name', 'N', 'Depression', 'Anxiety', 'Stress', 'Burnout', 'Description']].to_string(index=False))
    print()

    # ------------------------------------------------------------------------
    # TEST SET EVALUATION
    # ------------------------------------------------------------------------
    # Evaluate model generalization on held-out test data
    # Test set was never used during hyperparameter tuning or training
    # ------------------------------------------------------------------------
    print("Test-Set - 20% of the data was kept aside and used for testing")
    print("="*70)
    print("Test set was held out during hyperparameter tuning - now evaluating final model")
    print("="*70)

    print("Encoding test data with trained autoencoder...")
    test_dataset = TensorDataset(test_tensor.cpu())
    test_loader = DataLoader(test_dataset, batch_size=best_batch_size, shuffle=False)

    final_model.eval()
    test_latent_vectors = []
    test_reconstructions = []


    with torch.no_grad():
        for batch_data in test_loader:
            data = batch_data[0].to(device)
            latent = final_model.encoder(data)
            reconstructed = final_model(data)
            test_latent_vectors.append(latent.cpu().numpy())
            test_reconstructions.append(reconstructed.cpu().numpy())
    test_latent = np.vstack(test_latent_vectors)
    test_recon = np.vstack(test_reconstructions)


    print(f"  Test latent vectors: {test_latent.shape}")
    print(f"  Test reconstructions: {test_recon.shape}")

    # Compute test reconstruction error
    test_data_np = test_tensor.cpu().numpy()
    test_recon_loss = np.mean((test_data_np - test_recon) ** 2)
    print(f"  Test reconstruction loss (MSE): {test_recon_loss:.6f}")

    # Assign test samples to clusters using trained centroids
    # CRITICAL: Use centroids from train+val, don't retrain K-means
    # This tests if cluster structure generalizes to new data
    print(f"\nAssigning test samples to clusters...")
    from scipy.spatial.distance import cdist
    test_distances = cdist(test_latent, cluster_centroids, metric='euclidean')
    test_cluster_assignments = np.argmin(test_distances, axis=1)

    print(f"  Test cluster assignments: {Counter(test_cluster_assignments)}")

    # Evaluate clustering quality on test set
    # Check if we have at least 2 clusters (required for silhouette score)
    n_unique_test_clusters = len(np.unique(test_cluster_assignments))
    if n_unique_test_clusters < 2:
        print(f"⚠️ ERROR: Test clustering produced only {n_unique_test_clusters} cluster(s). Cannot compute metrics.")
        test_sil_score = -1
        test_ch_score = 0
        test_db_score = float('inf')
    else:
        test_sil_score = silhouette_score(test_latent, test_cluster_assignments)
        test_ch_score = calinski_harabasz_score(test_latent, test_cluster_assignments)
        test_db_score = davies_bouldin_score(test_latent, test_cluster_assignments)

    print(f"\nTest Set Clustering Quality:")
    print(f"  Silhouette Score: {test_sil_score:.6f}")
    print(f"  Calinski-Harabasz: {test_ch_score:.6f}")
    print(f"  Davies-Bouldin: {test_db_score:.6f}")

    # Check for overfitting: Compare train+val vs test performance
    if final_sil_score != 0:
        sil_diff_pct = abs(final_sil_score - test_sil_score) / final_sil_score * 100
    else:
        sil_diff_pct = float('inf')  # Handle edge case where final_sil_score is 0
    if sil_diff_pct < 5:
        print(f"\n✓ Good generalization: Test performance within {sil_diff_pct:.2f}% of train+val")
    elif sil_diff_pct < 10:
        print(f"\n⚠ Moderate generalization gap: Test performance {sil_diff_pct:.2f}% different from train+val")
    else:
        print(f"\n✗ Potential overfitting: Test performance {sil_diff_pct:.2f}% different from train+val")

    print("="*70 + "\n")

    # ------------------------------------------------------------------------
    # VISUALIZATIONS
    # ------------------------------------------------------------------------
    # Create visualizations of latent space and profile characteristics
    # ------------------------------------------------------------------------
    print("Visualizing Latent Space and Profile Characteristics")

    #1. Latent Space Visualization
    print("\n1. Latent Space Viz")

    if best_latent_dim == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(latent_vectors_all[:, 0], latent_vectors_all[:, 1], 
                            c=cluster_labels_all, cmap='viridis', alpha=0.6, s=30)
        ax.scatter(cluster_centroids[:, 0], cluster_centroids[:, 1], c='red', marker='x', 
                  s=300, linewidths=4, label='Cluster Centroids', zorder=5)
        #Add profile labels to centroids

        for i, (x,y) in enumerate(cluster_centroids):
            profile_name = profile_summary[i].get('Profile_Name', f'P{i+1}')
            ax.annotate(profile_name, (x,y), xytext=(5,5), textcoords='offset points',
                       fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3',
                       facecolor='yellow', alpha=0.7))

        ax.set_xlabel('Latent Dimension 1', fontsize=12)
        ax.set_ylabel('Latent Dimension 2', fontsize=12)
        ax.set_title('Latent Space Visualization (Colored by Profile)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Profile')
        plt.tight_layout()
        plt.show()

    elif best_latent_dim == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        scatter = ax.scatter(latent_vectors_all[:, 0], latent_vectors_all[:, 1], latent_vectors_all[:, 2],
                            c=cluster_labels_all, cmap='viridis', alpha=0.6, s=30)
        ax.scatter(cluster_centroids[:, 0], cluster_centroids[:, 1], cluster_centroids[:, 2],
                  c='red', marker='x', s=300, linewidths=4, label='Centroids')

        ax.set_xlabel('Latent Dim 1', fontsize=12)
        ax.set_ylabel('Latent Dim 2', fontsize=12)
        ax.set_zlabel('Latent Dim 3', fontsize=12)
        ax.set_title('Latent Space Visualization (3D)', fontsize=14, fontweight='bold')
        ax.legend()
        plt.colorbar(scatter, ax=ax, label='Profile')
        plt.tight_layout()
        plt.show()

    #2. Profile Characteristics Heatmap
    print("\n2. Profile Characteristics Heatmap:")
    profile_matrix = pd.DataFrame(profile_summary)[['Profile', 'Depression', 'Anxiety', 'Stress', 'Burnout']].set_index('Profile')
    profile_matrix_normalized = (profile_matrix - profile_matrix.min()) / (profile_matrix.max() - profile_matrix.min())

    plt.figure(figsize=(10, 6))
    sns.heatmap(profile_matrix_normalized.T, annot=profile_matrix.T, fmt='.3f', cmap='RdYlGn_r',
               cbar_kws={'label': 'Normalized Symptom Level'}, linewidths=0.5, linecolor='black')
    plt.title('Profile Characteristics Heatmap\n(Normalized Symptom Levels)', fontsize=14, fontweight='bold')
    plt.xlabel('Profile', fontsize=12)
    plt.ylabel('Symptom', fontsize=12)
    plt.tight_layout()
    plt.show()


    #Profile Bar Chart
    print("\n3. Profile Symptom Levels (Bar Chart):")
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(profile_summary))
    width = 0.2

    for i, profile in enumerate(profile_summary):
        ax.bar(x[i] - 1.5*width, profile['Depression'], width, label='Depression' if i == 0 else '', color='#ff6b6b')
        ax.bar(x[i] - 0.5*width, profile['Anxiety'], width, label='Anxiety' if i == 0 else '', color='#4ecdc4')
        ax.bar(x[i] + 0.5*width, profile['Stress'], width, label='Stress' if i == 0 else '', color='#45b7d1')
        ax.bar(x[i] + 1.5*width, profile['Burnout'], width, label='Burnout' if i == 0 else '', color='#f9ca24')

    ax.set_xlabel('Profile', fontsize=12)
    ax.set_ylabel('Symptom Level', fontsize=12)
    ax.set_title('Symptom Levels by Profile', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([p['Profile'] for p in profile_summary])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


    #Cluster Size Distribution
    print("\n4. Cluster Size Distribution:")

    fig, ax = plt.subplots(figsize=(8, 6))
    cluster_sizes = [p['N'] for p in profile_summary]
    cluster_labels_viz = [p['Profile'] for p in profile_summary]
    colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_sizes)))

    bars = ax.bar(cluster_labels_viz, cluster_sizes, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Profile', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Profile Size Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()





    # ------------------------------------------------------------------------
    # H3 VALIDATION: Clinical Utility Testing
    # ------------------------------------------------------------------------
    # Test if profile membership predicts therapy utilization
    # Only available for D1-Swiss dataset
    # ------------------------------------------------------------------------
    if dataset_name == "D1-Swiss" and "PSYT_Therapy_Use" in all_data_df.columns:
        print("\nH3 VALIDATION: Testing Clinical Utility of Profiles")
        print("="*70)
        print("Hypothesis H3: Profile membership is associated with therapy utilization")
        print("Using FULL dataset (train+val+test) for maximum statistical power")
        print("="*70)

        y_therapy = all_data_df["PSYT_Therapy_Use"].values
        train_val_therapy, test_therapy = train_test_split(
            y_therapy,
            test_size=0.2,
            random_state=RANDOM_SEED,
        )
        y_therapy_aligned = np.concatenate([train_val_therapy, test_therapy])
        all_cluster_labels = np.concatenate([cluster_labels_all, test_cluster_assignments])

        assert len(y_therapy_aligned) == len(all_cluster_labels), "Misalignment"
        print(f"\n✓ Data aligned: {len(all_cluster_labels)} samples")
        print(f"  Therapy use rate: {y_therapy_aligned.mean():.2%} ({y_therapy_aligned.sum()}/{len(y_therapy_aligned)})")

        print("\nChi-Square Test for Independence:")
        print("="*70)

        contingency = pd.crosstab(all_cluster_labels, y_therapy_aligned)
        chi2, p, dof, expected = chi2_contingency(contingency)

        print("   Contingency Table:")
        print(contingency)
        print(f"\n   Chi-square statistic: χ² = {chi2:.4f}")
        print(f"   Degrees of freedom: df = {dof}")
        print(f"   p-value: p = {p:.6f}")

        alpha = 0.05
        if p < alpha:
            print(f"\n   ✓ SIGNIFICANT (p < {alpha}): Profile membership IS associated with therapy utilization")
            print("   → H3 VALIDATED: Profiles have clinical utility")
        else:
            print(f"\n   ✗ NOT SIGNIFICANT (p >= {alpha}): No association detected")
            print("   → H3 NOT VALIDATED")

        print("\nCramer V Effect Size:")
        print("="*70)
        n = contingency.values.sum()
        min_dim = min(contingency.shape)
        cramers_v = np.sqrt(chi2 / (n * (min_dim - 1)))
        print(f"   Cramer's V: {cramers_v:.4f}")

        if cramers_v < 0.10:
            effect_size = "negligible"
        elif cramers_v < 0.30:
            effect_size = "small"
        elif cramers_v < 0.50:
            effect_size = "medium"
        else:
            effect_size = "large"
        print(f"   Effect size: {effect_size}")

        print("\n3. Post-Hoc Analysis: Standardized Residuals:")
        print("-"*70)
        print("   (Values > |2| indicate significant deviation from expected)")
        residuals = (contingency.values - expected) / np.sqrt(expected + 1e-10)
        residuals_df = pd.DataFrame(
            residuals,
            index=[f'P{k+1}' for k in range(best_k)],
            columns=['No Therapy', 'Therapy']
        )
        print(residuals_df.round(3))

        print("\n   Significant deviations:")
        for i in range(best_k):
            for j in range(2):
                if abs(residuals[i, j]) > 2:
                    profile_name = profile_summary[i].get('Profile_Name', f'P{i+1}')
                    therapy_status = 'Therapy' if j == 1 else 'No Therapy'
                    direction = 'Higher' if residuals[i, j] > 0 else 'Lower'
                    print(f"      • {profile_name} - {therapy_status}: {direction} than expected (residual = {residuals[i, j]:.2f})")

        print("\n" + "="*70)
        print("H3 VALIDATION SUMMARY:")
        print("="*70)
        print(f"Dataset: {dataset_name} (N={len(all_cluster_labels)})")
        print(f"Chi-square: χ² = {chi2:.4f}, p = {p:.6f}, df = {dof}")
        print(f"Cramér's V = {cramers_v:.4f} ({effect_size} effect)")
        print(f"H3 Status: {' VALIDATED' if p < alpha else '✗ NOT VALIDATED'}")

        if p < alpha:
            print("\nConclusion:")
            print("  Profiles demonstrate clinical utility by predicting therapy utilization.")
            print("  This supports the use of these profiles for targeted mental health interventions.")

        print("="*70 + "\n")
    else:
        print("\nSkipping H3 validation (only available for D1-Swiss with PSYT_Therapy_Use)")

    result = {
        'dataset_name': dataset_name,
        'train_val_data': train_val_data,
        'latent_vectors_all': latent_vectors_all,
        'cluster_labels_all': cluster_labels_all,
        'cluster_centroids': cluster_centroids,
        'best_k': best_k,
        'best_latent_dim': best_latent_dim,
        'final_sil_score': final_sil_score,
        'final_ch_score': final_ch_score,
        'final_db_score': final_db_score,
        'reconstruction_loss': test_recon_loss,  # Store test reconstruction loss for comparison
    }
    PIPELINE_RESULTS[dataset_name] = result
    return result


# In[ ]:


# ============================================================================
# CELL 3: TRAINING AND VALIDATION FUNCTION
# ============================================================================
# Purpose: Train autoencoder and evaluate clustering quality in latent space
# Key feature: Model quality evaluated by clustering metrics, not just
#              reconstruction loss, because goal is finding distinct profiles
# ============================================================================

def train_and_validate_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, device):
    """
    Train and validate the autoencoder model and evaluate reconstruction accuracy and latent quality.

    Uses multiple validation methods for K selection with consensus/voting approach:
    - Silhouette score (primary, tiebreaker)
    - Calinski-Harabasz index
    - Davies-Bouldin index
    - Elbow method (WCSS-based knee detection)

    K selection: Uses consensus voting - if 2+ methods agree on K, that K is selected.
    If no consensus, falls back to silhouette score (most interpretable).

    Returns:
        train_losses: list of training loss values
        val_losses: list of validation loss values
        optimal_k: optimal number of clusters (consensus or silhouette-based)
        best_silhouette_score: best silhouette score achieved
        latent_vectors: latent representations (train+val combined)
        validation_metrics: dict with all K validation metrics and consensus info
    """
    model.train()
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training phase
        epoch_training_loss = 0.0
        for batch_data in train_loader:
            batch_data = batch_data[0].to(device)  # Unpack tuple from DataLoader
            optimizer.zero_grad()
            reconstructed = model(batch_data)
            loss = criterion(reconstructed, batch_data)  # MSE Reconstruction Loss
            loss.backward()
            optimizer.step()
            epoch_training_loss += loss.item()
        avg_train_loss = epoch_training_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        epoch_validation_loss = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                batch_data = batch_data[0].to(device)
                reconstructed = model(batch_data)
                loss = criterion(reconstructed, batch_data)
                epoch_validation_loss += loss.item()
        avg_val_loss = epoch_validation_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        model.train()

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    # ------------------------------------------------------------------------
    # EXTRACT LATENT VECTORS AND EVALUATE CLUSTERING QUALITY
    # ------------------------------------------------------------------------
    # After training, extract latent representations and evaluate clustering
    # ------------------------------------------------------------------------
    model.eval()
    all_latent_vectors = []

    with torch.no_grad():
        # Extract latent vectors from training data
        for batch_data in train_loader:
            data = batch_data[0].to(device)
            latent = model.encoder(data)
            all_latent_vectors.append(latent.cpu().numpy())

        # Extract latent vectors from validation data
        for batch_data in val_loader:
            data = batch_data[0].to(device)
            latent = model.encoder(data)
            all_latent_vectors.append(latent.cpu().numpy())

    # Combine all latent vectors
    latent_vectors = np.vstack(all_latent_vectors)

    # DIAGNOSTIC: Check if model collapsed
    print(f"\n=== LATENT SPACE DIAGNOSTIC ===")
    print(f"Shape: {latent_vectors.shape}")
    print(f"Mean per dimension: {latent_vectors.mean(axis=0)}")
    print(f"Std per dimension: {latent_vectors.std(axis=0)}")
    print(f"Min per dimension: {latent_vectors.min(axis=0)}")
    print(f"Max per dimension: {latent_vectors.max(axis=0)}")
    print(f"Overall std: {latent_vectors.std():.6f}")
    if latent_vectors.std() < 0.01:
        print("⚠️ MODEL COLLAPSED: All latent vectors are nearly identical!")
        print("   This means the encoder isn't learning useful representations.")
    print("="*70)

    # =========================================================================
    # K Selection via Multiple Validation Methods (Convergent Validity)
    # =========================================================================
    k_range = range(2, 7)  # K = 2, 3, 4, 5, 6

    # Initialize metric lists
    silhouette_scores_list = []
    calinski_harabasz_scores = []
    davies_bouldin_scores = []
    wcss_values = []  # Within-cluster sum of squares (for elbow method)

    best_silhouette_score = -1

    print(f"\nTesting K values: {list(k_range)}")
    print("="*70)

    for k in k_range:
        print(f"K={k}: Running K-means...", end=" ", flush=True)
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        cluster_labels = kmeans.fit_predict(latent_vectors)
        print("Done. ", end="", flush=True)

        # Check if we have at least 2 clusters (required for silhouette score)
        n_unique_clusters = len(np.unique(cluster_labels))
        if n_unique_clusters < 2:
            print(f"Warning: Only {n_unique_clusters} cluster(s). Skipping metrics.")
            # Assign dummy values that will be ignored
            sil_score = -1
            ch_score = 0
            db_score = float('inf')
        else:
            print("Computing metrics...", end=" ", flush=True)
            # Compute all validation metrics
            sil_score = silhouette_score(latent_vectors, cluster_labels)
            ch_score = calinski_harabasz_score(latent_vectors, cluster_labels)
            db_score = davies_bouldin_score(latent_vectors, cluster_labels)
            print(f"Sil={sil_score:.4f}, CH={ch_score:.2f}, DB={db_score:.4f}")
        wcss = kmeans.inertia_  # Within-cluster sum of squares

        # Store metrics
        silhouette_scores_list.append(sil_score)
        calinski_harabasz_scores.append(ch_score)
        davies_bouldin_scores.append(db_score)
        wcss_values.append(wcss)

        # Track best silhouette score
        if sil_score > best_silhouette_score:
            best_silhouette_score = sil_score

    print("="*70)
    print(f"K-means evaluation complete. Best silhouette score: {best_silhouette_score:.4f}")
    print("\nDetermining optimal K using consensus voting...")

    # =========================================================================
    # Consensus/Voting Approach for K Selection
    # =========================================================================
    # Determine optimal K from each method
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores_list)]
    optimal_k_ch = k_range[np.argmax(calinski_harabasz_scores)]  # Highest CH = best
    optimal_k_db = k_range[np.argmin(davies_bouldin_scores)]  # Lowest DB = best

    # Elbow Method: Find the "knee" where WCSS decrease rate slows down
    # Compute percentage decrease in WCSS for each K
    wcss_decreases = []
    for i in range(1, len(wcss_values)):
        if wcss_values[i-1] > 0:  # Avoid division by zero
            pct_decrease = ((wcss_values[i-1] - wcss_values[i]) / wcss_values[i-1]) * 100
            wcss_decreases.append(pct_decrease)
        else:
            wcss_decreases.append(0)

    # Find elbow: K where decrease rate drops most (knee point)
    # The elbow is where adding more clusters doesn't significantly reduce WCSS
    if len(wcss_decreases) > 0:
        decrease_rates = np.array(wcss_decreases)
        # Method: Find where the decrease rate drops most (elbow detection)
        # Compute the rate of change of decrease rates (second derivative of WCSS)
        if len(decrease_rates) > 1:
            # Rate changes: how much the decrease rate changes between consecutive K
            rate_changes = np.diff(decrease_rates)
            # Elbow is where rate change is most negative (biggest drop in decrease rate)
            # This means: decrease rate was high, then dropped significantly
            elbow_idx = np.argmin(rate_changes) + 1  # +1 because diff reduces length by 1
            # Ensure index is within valid range
            elbow_idx = min(elbow_idx, len(k_range) - 1)
            optimal_k_elbow = k_range[elbow_idx]
        else:
            # Fallback: use K with smallest decrease (conservative)
            min_decrease_idx = np.argmin(wcss_decreases) + 1
            min_decrease_idx = min(min_decrease_idx, len(k_range) - 1)
            optimal_k_elbow = k_range[min_decrease_idx]
    else:
        # Fallback to silhouette if WCSS calculation fails
        optimal_k_elbow = optimal_k_silhouette

    # Consensus voting: If 2+ methods agree, use that K
    # Note: Multiple comparisons across 4 metrics and 5 K values (K=2-6) are exploratory
    # We use consensus voting rather to find the optimal K

    k_votes = [optimal_k_silhouette, optimal_k_ch, optimal_k_db, optimal_k_elbow]
    k_counts = Counter(k_votes)
    most_common_k, consensus_count = k_counts.most_common(1)[0]

    if consensus_count >= 2:
        # Consensus reached: 2+ methods agree
        optimal_k = most_common_k
        consensus_status = f"Consensus: {consensus_count} methods agree on K={optimal_k}"
        consensus_reached = True
    else:
        # No consensus: fallback to silhouette (most interpretable)
        optimal_k = optimal_k_silhouette
        consensus_status = f"No consensus (Sil={optimal_k_silhouette}, CH={optimal_k_ch}, DB={optimal_k_db}, Elbow={optimal_k_elbow}). Using silhouette K={optimal_k}"
        consensus_reached = False

    print(f"✓ Optimal K selected: {optimal_k} ({consensus_status})")
    print("="*70)

    # Package all validation metrics for analysis
    validation_metrics = {
        'k_values': list(k_range),
        'silhouette_scores': silhouette_scores_list,
        'calinski_harabasz_scores': calinski_harabasz_scores,
        'davies_bouldin_scores': davies_bouldin_scores,
        'wcss_values': wcss_values,
        'wcss_decreases': wcss_decreases if len(wcss_decreases) > 0 else [],
        'optimal_k_silhouette': optimal_k_silhouette,
        'optimal_k_ch': optimal_k_ch,
        'optimal_k_db': optimal_k_db,
        'optimal_k_elbow': optimal_k_elbow,
        'optimal_k_consensus': optimal_k,
        'consensus_reached': consensus_reached,
        'consensus_status': consensus_status,
        'k_votes': k_votes
    }

    return train_losses, val_losses, optimal_k, best_silhouette_score, latent_vectors, validation_metrics


# In[ ]:


# ============================================================================
# NOTE: This cell previously contained duplicate hyperparameter tuning code
# that was causing NameError because kfold and train_val_tensor are only
# defined inside run_autoencoder_pipeline() function scope.
#
# The hyperparameter tuning code is now properly contained within 
# run_autoencoder_pipeline() function.
#
# To run the pipeline, use:
#   run_autoencoder_pipeline("D1-Swiss")
# ============================================================================
pass


# In[ ]:


# =========================================================================
# PCA Comparison: Justify Autoencoder Choice
# =========================================================================

# LEGACY VERSION (kept for reference):
"""
SWISS_DATASET = "D1-Swiss"

if SWISS_DATASET not in PIPELINE_RESULTS:
    print(f"{SWISS_DATASET} not yet processed. Running autoencoder pipeline once for PCA comparison...")
    run_autoencoder_pipeline(SWISS_DATASET)

swiss_results = PIPELINE_RESULTS[SWISS_DATASET]
train_val_data = swiss_results['train_val_data']
latent_vectors_all = swiss_results['latent_vectors_all']
cluster_labels_all = swiss_results['cluster_labels_all']
cluster_centroids = swiss_results['cluster_centroids']
best_k = swiss_results['best_k']
best_latent_dim = swiss_results['best_latent_dim']
final_sil_score = swiss_results['final_sil_score']
final_ch_score = swiss_results['final_ch_score']
final_db_score = swiss_results['final_db_score']

print("METHOD VALIDATION: PCA vs Autoencoder Comparison")
print("="*70)
print("Testing if autoencoder captures nonlinear patterns better than PCA")
print("Testing PCA with all possible dimensions (1-4) to find optimal PCA")
print("="*70)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Prepare data
scaler = StandardScaler()
train_val_scaled = scaler.fit_transform(train_val_data)

# Test PCA with all possible dimensions (1, 2, 3, 4)
print("\n1. Testing PCA with all dimensions:")
pca_results = []

for pca_dim in range(1, 5):  # 1, 2, 3, 4
    # Fit PCA
    pca = PCA(n_components=pca_dim)
    pca_latent = pca.fit_transform(train_val_scaled)
    explained_var = pca.explained_variance_ratio_.sum()

    # Cluster with optimal K (same as autoencoder)
    pca_kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    pca_cluster_labels = pca_kmeans.fit_predict(pca_latent)

    # Evaluate clustering quality
    pca_sil_score = silhouette_score(pca_latent, pca_cluster_labels)
    pca_ch_score = calinski_harabasz_score(pca_latent, pca_cluster_labels)
    pca_db_score = davies_bouldin_score(pca_latent, pca_cluster_labels)

    pca_results.append({
        'dim': pca_dim,
        'explained_var': explained_var,
        'silhouette': pca_sil_score,
        'calinski_harabasz': pca_ch_score,
        'davies_bouldin': pca_db_score
    })

    print(f"   PCA dim={pca_dim}: Sil={pca_sil_score:.6f}, CH={pca_ch_score:.6f}, DB={pca_db_score:.6f}, Var={explained_var:.4f}")

# Find best PCA configuration (by silhouette score)
pca_results_df = pd.DataFrame(pca_results)
best_pca_idx = pca_results_df['silhouette'].idxmax()
best_pca_config = pca_results_df.iloc[best_pca_idx]

print(f"\n   Best PCA: dim={int(best_pca_config['dim'])}, Sil={best_pca_config['silhouette']:.6f}")

# Get best PCA latent vectors for visualization
best_pca_dim = int(best_pca_config['dim'])
best_pca = PCA(n_components=best_pca_dim)
pca_latent_best = best_pca.fit_transform(train_val_scaled)
pca_kmeans_best = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
pca_cluster_labels_best = pca_kmeans_best.fit_predict(pca_latent_best)
pca_centroids_best = pca_kmeans_best.cluster_centers_

# Autoencoder results (from Cell 7)
print("\n2. Autoencoder (Nonlinear Method):")
print(f"   AE latent dim: {best_latent_dim}")
print(f"   AE latent vectors shape: {latent_vectors_all.shape}")
print(f"   AE Clustering Quality:")
print(f"     Silhouette Score: {final_sil_score:.6f}")
print(f"     Calinski-Harabasz: {final_ch_score:.6f}")
print(f"     Davies-Bouldin: {final_db_score:.6f}")

# Comparison: Best PCA vs Autoencoder
print("\n3. Comparison (Best PCA vs Autoencoder):")
best_pca_sil = best_pca_config['silhouette']
best_pca_ch = best_pca_config['calinski_harabasz']
best_pca_db = best_pca_config['davies_bouldin']

sil_improvement = ((final_sil_score - best_pca_sil) / best_pca_sil) * 100
ch_improvement = ((final_ch_score - best_pca_ch) / best_pca_ch) * 100
db_improvement = ((best_pca_db - final_db_score) / best_pca_db) * 100  # DB: lower is better

print(f"   Silhouette: AE {final_sil_score:.6f} vs Best PCA {best_pca_sil:.6f} ({sil_improvement:+.2f}%)")
print(f"   Calinski-Harabasz: AE {final_ch_score:.6f} vs Best PCA {best_pca_ch:.6f} ({ch_improvement:+.2f}%)")
print(f"   Davies-Bouldin: AE {final_db_score:.6f} vs Best PCA {best_pca_db:.6f} ({db_improvement:+.2f}% improvement)")

# Conclusion
print("\n4. Conclusion:")
if final_sil_score > best_pca_sil:
    print(f"   ✓ Autoencoder achieves {sil_improvement:.2f}% better silhouette score than best PCA")
    print(f"   → Suggests latent structure contains nonlinear patterns")
    print(f"   → Justifies use of autoencoder over linear PCA")
else:
    print(f"   Note: Best PCA (dim={best_pca_dim}) performs similarly ({best_pca_sil:.6f} vs {final_sil_score:.6f})")
    print(f"   → Linear patterns may be sufficient, but AE provides flexibility")

print("="*70 + "\n")

# Visualization: Compare latent spaces
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

if best_pca_dim == 2:
    scatter1 = axes[0].scatter(pca_latent_best[:, 0], pca_latent_best[:, 1], c=pca_cluster_labels_best, 
                              cmap='viridis', alpha=0.6, s=20)
    axes[0].scatter(pca_centroids_best[:, 0], pca_centroids_best[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
    axes[0].set_xlabel('PC1', fontsize=12)
    axes[0].set_ylabel('PC2', fontsize=12)
    axes[0].set_title(f'Best PCA (dim={best_pca_dim}, Linear)\nSilhouette: {best_pca_sil:.4f}', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    scatter2 = axes[1].scatter(latent_vectors_all[:, 0], latent_vectors_all[:, 1], c=cluster_labels_all,
                              cmap='viridis', alpha=0.6, s=20)
    axes[1].scatter(cluster_centroids[:, 0], cluster_centroids[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
    axes[1].set_xlabel('Latent Dim 1', fontsize=12)
    axes[1].set_ylabel('Latent Dim 2', fontsize=12)
    axes[1].set_title(f'Autoencoder (dim={best_latent_dim}, Nonlinear)\nSilhouette: {final_sil_score:.4f}', 
                     fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.colorbar(scatter1, ax=axes[0], label='Cluster')
    plt.colorbar(scatter2, ax=axes[1], label='Cluster')
elif best_pca_dim == 3 or best_latent_dim == 3:
    from mpl_toolkits.mplot3d import Axes3D
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    if best_pca_dim >= 3:
        ax1.scatter(pca_latent_best[:, 0], pca_latent_best[:, 1], pca_latent_best[:, 2], 
                   c=pca_cluster_labels_best, cmap='viridis', alpha=0.6, s=20)
        ax1.scatter(pca_centroids_best[:, 0], pca_centroids_best[:, 1], pca_centroids_best[:, 2], 
                   c='red', marker='x', s=200, linewidths=3)
    else:
        pca_3d = np.zeros((len(pca_latent_best), 3))
        pca_3d[:, :best_pca_dim] = pca_latent_best
        ax1.scatter(pca_3d[:, 0], pca_3d[:, 1], pca_3d[:, 2], 
                   c=pca_cluster_labels_best, cmap='viridis', alpha=0.6, s=20)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    ax1.set_title(f'Best PCA (dim={best_pca_dim})')

    if best_latent_dim >= 3:
        ax2.scatter(latent_vectors_all[:, 0], latent_vectors_all[:, 1], latent_vectors_all[:, 2], 
                   c=cluster_labels_all, cmap='viridis', alpha=0.6, s=20)
        ax2.scatter(cluster_centroids[:, 0], cluster_centroids[:, 1], cluster_centroids[:, 2], 
                   c='red', marker='x', s=200, linewidths=3)
    else:
        ae_3d = np.zeros((len(latent_vectors_all), 3))
        ae_3d[:, :best_latent_dim] = latent_vectors_all
        ax2.scatter(ae_3d[:, 0], ae_3d[:, 1], ae_3d[:, 2], 
                   c=cluster_labels_all, cmap='viridis', alpha=0.6, s=20)
        ax2.scatter(cluster_centroids[:, 0], cluster_centroids[:, 1], np.zeros(len(cluster_centroids)), 
                   c='red', marker='x', s=200, linewidths=3)
    ax2.set_xlabel('Latent Dim 1')
    ax2.set_ylabel('Latent Dim 2')
    ax2.set_zlabel('Latent Dim 3')
    ax2.set_title(f'Autoencoder (dim={best_latent_dim})\nSil: {final_sil_score:.4f} vs PCA: {best_pca_sil:.4f}')
else:
    scatter1 = axes[0].scatter(pca_latent_best[:, 0], np.zeros(len(pca_latent_best)), 
                              c=pca_cluster_labels_best, cmap='viridis', alpha=0.6, s=20)
    axes[0].scatter(pca_centroids_best[:, 0], np.zeros(len(pca_centroids_best)), 
                   c='red', marker='x', s=200, linewidths=3, label='Centroids')
    axes[0].set_xlabel('PC1', fontsize=12)
    axes[0].set_ylabel('(Projected)', fontsize=12)
    axes[0].set_title(f'Best PCA (dim={best_pca_dim})', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    scatter2 = axes[1].scatter(latent_vectors_all[:, 0], np.zeros(len(latent_vectors_all)) if best_latent_dim == 1 else latent_vectors_all[:, 1], 
                              c=cluster_labels_all, cmap='viridis', alpha=0.6, s=20)
    axes[1].scatter(cluster_centroids[:, 0], np.zeros(len(cluster_centroids)) if best_latent_dim == 1 else cluster_centroids[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='Centroids')
    axes[1].set_xlabel('Latent Dim 1', fontsize=12)
    axes[1].set_ylabel('Latent Dim 2' if best_latent_dim >= 2 else '(Projected)', fontsize=12)
    axes[1].set_title(f'Autoencoder (dim={best_latent_dim})', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.colorbar(scatter1, ax=axes[0], label='Cluster')
    plt.colorbar(scatter2, ax=axes[1], label='Cluster')

plt.tight_layout()
plt.show()

# Summary table
print("\n5. Summary Table:")
comparison_df = pd.DataFrame({
    'Method': ['Best PCA', 'Autoencoder'],
    'Latent_Dim': [best_pca_dim, best_latent_dim],
    'Silhouette': [best_pca_sil, final_sil_score],
    'Calinski_Harabasz': [best_pca_ch, final_ch_score],
    'Davies_Bouldin': [best_pca_db, final_db_score]
})
print(comparison_df.to_string(index=False))
print()
"""

# ACTIVE VERSION (executes):
SWISS_DATASET = "D1-Swiss"

if SWISS_DATASET not in PIPELINE_RESULTS:
    print(f"{SWISS_DATASET} not yet processed. Running autoencoder pipeline once for PCA comparison...")
    run_autoencoder_pipeline(SWISS_DATASET)

swiss_results = PIPELINE_RESULTS[SWISS_DATASET]
train_val_data = swiss_results['train_val_data']
latent_vectors_all = swiss_results['latent_vectors_all']
cluster_labels_all = swiss_results['cluster_labels_all']
cluster_centroids = swiss_results['cluster_centroids']
best_k = swiss_results['best_k']
best_latent_dim = swiss_results['best_latent_dim']
final_sil_score = swiss_results['final_sil_score']
final_ch_score = swiss_results['final_ch_score']
final_db_score = swiss_results['final_db_score']

print("METHOD VALIDATION: PCA vs Autoencoder Comparison")
print("="*70)
print("Testing if autoencoder captures nonlinear patterns better than PCA")
print("Testing PCA with all possible dimensions (1-4) to find optimal PCA")
print("="*70)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Prepare data
scaler = StandardScaler()
train_val_scaled = scaler.fit_transform(train_val_data)

# Test PCA with all possible dimensions (1, 2, 3, 4)
print("\n1. Testing PCA with all dimensions:")
pca_results = []

for pca_dim in range(1, 5):  # 1, 2, 3, 4
    pca = PCA(n_components=pca_dim)
    pca_latent = pca.fit_transform(train_val_scaled)
    explained_var = pca.explained_variance_ratio_.sum()

    pca_kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    pca_cluster_labels = pca_kmeans.fit_predict(pca_latent)

    pca_sil_score = silhouette_score(pca_latent, pca_cluster_labels)
    pca_ch_score = calinski_harabasz_score(pca_latent, pca_cluster_labels)
    pca_db_score = davies_bouldin_score(pca_latent, pca_cluster_labels)

    pca_results.append({
        'dim': pca_dim,
        'explained_var': explained_var,
        'silhouette': pca_sil_score,
        'calinski_harabasz': pca_ch_score,
        'davies_bouldin': pca_db_score
    })

    print(f"   PCA dim={pca_dim}: Sil={pca_sil_score:.6f}, CH={pca_ch_score:.6f}, DB={pca_db_score:.6f}, Var={explained_var:.4f}")

pca_results_df = pd.DataFrame(pca_results)
best_pca_idx = pca_results_df['silhouette'].idxmax()
best_pca_config = pca_results_df.iloc[best_pca_idx]

print(f"\n   Best PCA: dim={int(best_pca_config['dim'])}, Sil={best_pca_config['silhouette']:.6f}")

best_pca_dim = int(best_pca_config['dim'])
best_pca = PCA(n_components=best_pca_dim)
pca_latent_best = best_pca.fit_transform(train_val_scaled)
pca_kmeans_best = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
pca_cluster_labels_best = pca_kmeans_best.fit_predict(pca_latent_best)
pca_centroids_best = pca_kmeans_best.cluster_centers_

print("\n2. Autoencoder (Nonlinear Method):")
print(f"   AE latent dim: {best_latent_dim}")
print(f"   AE latent vectors shape: {latent_vectors_all.shape}")
print("   AE Clustering Quality:")
print(f"     Silhouette Score: {final_sil_score:.6f}")
print(f"     Calinski-Harabasz: {final_ch_score:.6f}")
print(f"     Davies-Bouldin: {final_db_score:.6f}")

print("\n3. Comparison (Best PCA vs Autoencoder):")
best_pca_sil = best_pca_config['silhouette']
best_pca_ch = best_pca_config['calinski_harabasz']
best_pca_db = best_pca_config['davies_bouldin']

sil_improvement = ((final_sil_score - best_pca_sil) / best_pca_sil) * 100
ch_improvement = ((final_ch_score - best_pca_ch) / best_pca_ch) * 100
db_improvement = ((best_pca_db - final_db_score) / best_pca_db) * 100

print(f"   Silhouette: AE {final_sil_score:.6f} vs Best PCA {best_pca_sil:.6f} ({sil_improvement:+.2f}%)")
print(f"   Calinski-Harabasz: AE {final_ch_score:.6f} vs Best PCA {best_pca_ch:.6f} ({ch_improvement:+.2f}%)")
print(f"   Davies-Bouldin: AE {final_db_score:.6f} vs Best PCA {best_pca_db:.6f} ({db_improvement:+.2f}% improvement)")

print("\n4. Conclusion:")
if final_sil_score > best_pca_sil:
    print(f"   ✓ Autoencoder achieves {sil_improvement:.2f}% better silhouette score than best PCA")
    print(f"   → Suggests latent structure contains nonlinear patterns")
    print(f"   → Justifies use of autoencoder over linear PCA")
else:
    print(f"   Note: Best PCA (dim={best_pca_dim}) performs similarly ({best_pca_sil:.6f} vs {final_sil_score:.6f})")
    print(f"   → Linear patterns may be sufficient, but AE provides flexibility")

print("="*70 + "\n")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

if best_pca_dim == 2:
    scatter1 = axes[0].scatter(pca_latent_best[:, 0], pca_latent_best[:, 1], c=pca_cluster_labels_best,
                              cmap='viridis', alpha=0.6, s=20)
    axes[0].scatter(pca_centroids_best[:, 0], pca_centroids_best[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
    axes[0].set_xlabel('PC1', fontsize=12)
    axes[0].set_ylabel('PC2', fontsize=12)
    axes[0].set_title(f'Best PCA (dim={best_pca_dim}, Linear)\nSilhouette: {best_pca_sil:.4f}', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    scatter2 = axes[1].scatter(latent_vectors_all[:, 0], latent_vectors_all[:, 1], c=cluster_labels_all,
                              cmap='viridis', alpha=0.6, s=20)
    axes[1].scatter(cluster_centroids[:, 0], cluster_centroids[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
    axes[1].set_xlabel('Latent Dim 1', fontsize=12)
    axes[1].set_ylabel('Latent Dim 2', fontsize=12)
    axes[1].set_title(f'Autoencoder (dim={best_latent_dim}, Nonlinear)\nSilhouette: {final_sil_score:.4f}', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.colorbar(scatter1, ax=axes[0], label='Cluster')
    plt.colorbar(scatter2, ax=axes[1], label='Cluster')
elif best_pca_dim == 3 or best_latent_dim == 3:
    from mpl_toolkits.mplot3d import Axes3D
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    if best_pca_dim >= 3:
        ax1.scatter(pca_latent_best[:, 0], pca_latent_best[:, 1], pca_latent_best[:, 2],
                   c=pca_cluster_labels_best, cmap='viridis', alpha=0.6, s=20)
        ax1.scatter(pca_centroids_best[:, 0], pca_centroids_best[:, 1], pca_centroids_best[:, 2],
                   c='red', marker='x', s=200, linewidths=3)
    else:
        pca_3d = np.zeros((len(pca_latent_best), 3))
        pca_3d[:, :best_pca_dim] = pca_latent_best
        ax1.scatter(pca_3d[:, 0], pca_3d[:, 1], pca_3d[:, 2],
                   c=pca_cluster_labels_best, cmap='viridis', alpha=0.6, s=20)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    ax1.set_title(f'Best PCA (dim={best_pca_dim})')

    if best_latent_dim >= 3:
        ax2.scatter(latent_vectors_all[:, 0], latent_vectors_all[:, 1], latent_vectors_all[:, 2],
                   c=cluster_labels_all, cmap='viridis', alpha=0.6, s=20)
        ax2.scatter(cluster_centroids[:, 0], cluster_centroids[:, 1], cluster_centroids[:, 2],
                   c='red', marker='x', s=200, linewidths=3)
    else:
        ae_3d = np.zeros((len(latent_vectors_all), 3))
        ae_3d[:, :best_latent_dim] = latent_vectors_all
        ax2.scatter(ae_3d[:, 0], ae_3d[:, 1], ae_3d[:, 2],
                   c=cluster_labels_all, cmap='viridis', alpha=0.6, s=20)
        ax2.scatter(cluster_centroids[:, 0], cluster_centroids[:, 1], np.zeros(len(cluster_centroids)),
                   c='red', marker='x', s=200, linewidths=3)
    ax2.set_xlabel('Latent Dim 1')
    ax2.set_ylabel('Latent Dim 2')
    ax2.set_zlabel('Latent Dim 3')
    ax2.set_title(f'Autoencoder (dim={best_latent_dim})\nSil: {final_sil_score:.4f} vs PCA: {best_pca_sil:.4f}')
else:
    scatter1 = axes[0].scatter(pca_latent_best[:, 0], np.zeros(len(pca_latent_best)),
                              c=pca_cluster_labels_best, cmap='viridis', alpha=0.6, s=20)
    axes[0].scatter(pca_centroids_best[:, 0], np.zeros(len(pca_centroids_best)),
                   c='red', marker='x', s=200, linewidths=3, label='Centroids')
    axes[0].set_xlabel('PC1', fontsize=12)
    axes[0].set_ylabel('(Projected)', fontsize=12)
    axes[0].set_title(f'Best PCA (dim={best_pca_dim})', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    scatter2 = axes[1].scatter(latent_vectors_all[:, 0], np.zeros(len(latent_vectors_all)) if best_latent_dim == 1 else latent_vectors_all[:, 1],
                              c=cluster_labels_all, cmap='viridis', alpha=0.6, s=20)
    axes[1].scatter(cluster_centroids[:, 0], np.zeros(len(cluster_centroids)) if best_latent_dim == 1 else cluster_centroids[:, 1],
                   c='red', marker='x', s=200, linewidths=3, label='Centroids')
    axes[1].set_xlabel('Latent Dim 1', fontsize=12)
    axes[1].set_ylabel('Latent Dim 2' if best_latent_dim >= 2 else '(Projected)', fontsize=12)
    axes[1].set_title(f'Autoencoder (dim={best_latent_dim})', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.colorbar(scatter1, ax=axes[0], label='Cluster')
    plt.colorbar(scatter2, ax=axes[1], label='Cluster')

plt.tight_layout()
plt.show()

print("\n5. Summary Table:")
comparison_df = pd.DataFrame({
    'Method': ['Best PCA', 'Autoencoder'],
    'Latent_Dim': [best_pca_dim, best_latent_dim],
    'Silhouette': [best_pca_sil, final_sil_score],
    'Calinski_Harabasz': [best_pca_ch, final_ch_score],
    'Davies_Bouldin': [best_pca_db, final_db_score]
})
print(comparison_df.to_string(index=False))
print()


# In[ ]:


# =========================================================================
# REPLICATION TESTING: H1/H2 Hypotheses (D1-Swiss as Reference)
# =========================================================================
# Only matches latent_dim (required for comparison), K can differ
# Reuses existing run_autoencoder_pipeline() function
# =========================================================================

# Additional imports (already imported in Cell 0, but included for clarity)
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# Configuration constants
EXTERNAL_DATASETS = ["D2-Cultural", "D3-Academic", "D4-Tech"]
REPLICATION_THRESHOLD_H1 = 0.70
REPLICATION_THRESHOLD_H2 = 0.50

print("="*80)
print("REPLICATION TESTING: Testing Profile Generalizability")
print("="*80)
print("Hypothesis H1: Universal profiles (r > 0.70)")
print("Hypothesis H2: Context-specific profiles (r < 0.50)")
print("="*80)
print("Reference: D1-Swiss (only latent_dim must match for comparison)")
print("="*80)

# ========================================================================
# STEP 1: PROCESS D1-SWISS (REFERENCE DATASET)
# ========================================================================
print("\nStep 1: Processing D1-Swiss (reference dataset)...")
print("-"*80)

if "D1-Swiss" not in PIPELINE_RESULTS:
    print("Running pipeline on D1-Swiss...")
    run_autoencoder_pipeline("D1-Swiss")
else:
    print("D1-Swiss already processed ✓")

d1_results = PIPELINE_RESULTS["D1-Swiss"]
d1_latent_dim = d1_results['best_latent_dim']
d1_k = d1_results['best_k']
d1_centroids = d1_results['cluster_centroids']

print(f"\nD1-Swiss Reference Configuration:")
print(f"  Optimal K: {d1_k}")
print(f"  Latent dimension: {d1_latent_dim}D")
print(f"  Centroids shape: {d1_centroids.shape}")

# ========================================================================
# STEP 2: PROCESS EXTERNAL DATASETS AND RETUNE IF NEEDED
# ========================================================================
print(f"\nStep 2: Processing external datasets and retuning to {d1_latent_dim}D if needed...")
print("-"*80)

standardized_results = {}

for ext_dataset in EXTERNAL_DATASETS:
    print(f"\nProcessing {ext_dataset}...")

    # Run pipeline if not already processed
    if ext_dataset not in PIPELINE_RESULTS:
        print(f"  Running pipeline on {ext_dataset}...")
        run_autoencoder_pipeline(ext_dataset)

    ext_result = PIPELINE_RESULTS[ext_dataset]
    ext_latent_dim = ext_result['best_latent_dim']

    print(f"  Original latent dimensions: {ext_latent_dim}D")

    if ext_latent_dim == d1_latent_dim:
        print(f"  ✓ Latent dimension matches D1-Swiss ({d1_latent_dim}D)")
        standardized_results[ext_dataset] = ext_result.copy()
    else:
        print(f"  ⚠ Dimension mismatch: {ext_latent_dim}D ≠ {d1_latent_dim}D")
        print(f"  → Retuning to match D1-Swiss latent_dim ({d1_latent_dim}D)...")
        print(f"    (K can differ - only latent_dim needs to match for comparison)")

        retrained_result = run_autoencoder_pipeline(
            ext_dataset, 
            force_latent_dim=d1_latent_dim
        )

        original_recon_loss = ext_result.get('reconstruction_loss', None)

        if original_recon_loss is not None:
            retrained_recon_loss = retrained_result.get('reconstruction_loss', None)

            if retrained_recon_loss is not None:
                if original_recon_loss != 0:
                    info_loss_pct = (
                        (retrained_recon_loss - original_recon_loss) / original_recon_loss
                    ) * 100

                    print(f"\n  Information Loss Analysis ({ext_latent_dim}D → {d1_latent_dim}D):")
                    print(f"    Original recon loss ({ext_latent_dim}D): {original_recon_loss:.6f}")
                    print(f"    Retuned recon loss ({d1_latent_dim}D): {retrained_recon_loss:.6f}")
                    print(f"    Increase: {info_loss_pct:+.2f}% (higher = more information lost)")

                    retrained_result['info_loss_pct'] = info_loss_pct
                    retrained_result['original_recon_loss'] = original_recon_loss
                else:
                    print(f"\n  Information Loss Analysis ({ext_latent_dim}D → {d1_latent_dim}D):")
                    print(f"    Original recon loss ({ext_latent_dim}D): {original_recon_loss:.6f}")
                    print(f"    Retuned recon loss ({d1_latent_dim}D): {retrained_recon_loss:.6f}")
                    print(f"    Note: Cannot calculate percentage change (original loss is 0)")

        standardized_results[ext_dataset] = retrained_result

# ========================================================================
# STEP 3: COMPARE CENTROIDS ACROSS DATASETS
# ========================================================================
print(f"\n{'='*80}")
print(f"Step 3: Comparing profiles across datasets (all in {d1_latent_dim}D space)")
print(f"{'='*80}")

replication_results = []

for ext_dataset in EXTERNAL_DATASETS:
    print(f"\n{'='*80}")
    print(f"Comparing {ext_dataset} profiles to D1-Swiss")
    print(f"{'='*80}")

    ext_result = standardized_results[ext_dataset]
    ext_centroids = ext_result['cluster_centroids']
    ext_k = ext_result['best_k']

    print(f"  K: {ext_k} (D1-Swiss K: {d1_k})")
    print(f"  Centroids shape: {ext_centroids.shape}")

    # Re-cluster if K differs
    if ext_k != d1_k:
        print(f"  Re-clustering with K={d1_k} to match D1-Swiss for comparison...")
        ext_latent_vectors = ext_result['latent_vectors_all']
        ext_kmeans = KMeans(n_clusters=d1_k, random_state=RANDOM_SEED, n_init=10)
        ext_cluster_labels = ext_kmeans.fit_predict(ext_latent_vectors)
        ext_centroids = ext_kmeans.cluster_centers_
        ext_k = d1_k

    # Match centroids using Hungarian Algorithm
    distance_matrix = cdist(d1_centroids, ext_centroids, metric='euclidean')
    row_indices, col_indices = linear_sum_assignment(distance_matrix)
    ext_centroids_matched = ext_centroids[col_indices]

    # Calculate dimension-wise correlations
    # NOTE: Sample size is K (number of clusters), typically 2-6, which is small for correlation
    # P-values should be interpreted with caution. Correlation coefficient is the primary metric.
    dim_correlations = []
    dim_p_values = []

    for dim in range(d1_latent_dim):
        d1_dim = d1_centroids[:, dim]
        ext_dim = ext_centroids_matched[:, dim]
        correlation, p_value = pearsonr(d1_dim, ext_dim)
        dim_correlations.append(correlation)
        dim_p_values.append(p_value)

    # Calculate overall metrics
    avg_correlation = np.mean(dim_correlations)

    # Use pearsonr for overall correlation (consistent with dimension-wise calculation)
    # NOTE: Sample size = K × latent_dim, still small but better than dimension-wise
    d1_flat = d1_centroids.flatten()
    ext_flat = ext_centroids_matched.flatten()
    overall_correlation, overall_p_value = pearsonr(d1_flat, ext_flat)

    print(f"\n  Replication scores:")
    print(f"    ⚠ Note: Correlation computed on {d1_k} centroids (n={d1_k})")
    print(f"    ⚠ P-values should be interpreted with caution due to small sample size")
    for dim in range(d1_latent_dim):
        print(f"    Dim {dim+1} correlation: {dim_correlations[dim]:.4f} (p={dim_p_values[dim]:.6f})")
    print(f"    Average correlation: {avg_correlation:.4f}")
    print(f"    Overall correlation: {overall_correlation:.4f} (p={overall_p_value:.6f}, n={len(d1_flat)})")

    # Test hypotheses
    h1_supported = overall_correlation > REPLICATION_THRESHOLD_H1
    h2_supported = overall_correlation < REPLICATION_THRESHOLD_H2

    print(f"\n  Hypothesis Testing:")
    print(f"    H1: {'Supported' if h1_supported else 'Not Supported'}")
    print(f"    H2: {'Supported' if h2_supported else 'Not Supported'}")

    if h1_supported:
        print(f"    → Conclusion: Profiles are UNIVERSAL across D1-Swiss and {ext_dataset}")
    elif h2_supported:
        print(f"    → Conclusion: Profiles are CONTEXT-SPECIFIC (different patterns)")
    else:
        print(f"    → Conclusion: Moderate replication (0.50 ≤ r ≤ 0.70)")

    replication_results.append({
        'dataset': ext_dataset,
        'replication_score': overall_correlation,
        'avg_dim_correlation': avg_correlation,
        'p_value': overall_p_value,
        'h1_supported': h1_supported,
        'h2_supported': h2_supported,
        'info_loss_pct': ext_result.get('info_loss_pct', None),
    })

    # ====================================================================
    # VISUALIZATION
    # ====================================================================
    if d1_latent_dim == 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # D1-Swiss centroids
        axes[0].scatter(
            d1_centroids[:, 0],
            d1_centroids[:, 1],
            c=range(d1_k),
            cmap='viridis',
            s=200,
            edgecolors='black',
            linewidth=2,
            marker='o',
            label='D1-Swiss'
        )

        for i, (x, y) in enumerate(d1_centroids):
            axes[0].annotate(
                f'P{i+1}',
                (x, y),
                xytext=(5, 5),
                textcoords='offset points',
                fontweight='bold'
            )

        axes[0].set_xlabel('Latent Dim 1', fontsize=12)
        axes[0].set_ylabel('Latent Dim 2', fontsize=12)
        axes[0].set_title('D1-Swiss Centroids (Reference)', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # External dataset centroids
        axes[1].scatter(
            ext_centroids_matched[:, 0],
            ext_centroids_matched[:, 1],
            c=range(ext_k),
            cmap='viridis',
            s=200,
            edgecolors='red',
            linewidth=2,
            marker='s',
            label=ext_dataset
        )

        for i, (x, y) in enumerate(ext_centroids_matched):
            axes[1].annotate(
                f'P{i+1}',
                (x, y),
                xytext=(5, 5),
                textcoords='offset points',
                fontweight='bold'
            )

        axes[1].set_xlabel('Latent Dim 1', fontsize=12)
        axes[1].set_ylabel('Latent Dim 2', fontsize=12)
        axes[1].set_title(
            f'{ext_dataset} Centroids (Matched)\nr = {overall_correlation:.4f}',
            fontsize=14,
            fontweight='bold'
        )
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    elif d1_latent_dim == 3:
        fig = plt.figure(figsize=(14, 6))

        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(
            d1_centroids[:, 0],
            d1_centroids[:, 1],
            d1_centroids[:, 2],
            c=range(d1_k),
            cmap='viridis',
            s=200,
            edgecolors='black',
            linewidth=2
        )
        ax1.set_xlabel('Latent Dim 1')
        ax1.set_ylabel('Latent Dim 2')
        ax1.set_zlabel('Latent Dim 3')
        ax1.set_title('D1-Swiss Centroids (Reference)')

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(
            ext_centroids_matched[:, 0],
            ext_centroids_matched[:, 1],
            ext_centroids_matched[:, 2],
            c=range(ext_k),
            cmap='viridis',
            s=200,
            edgecolors='red',
            linewidth=2
        )
        ax2.set_xlabel('Latent Dim 1')
        ax2.set_ylabel('Latent Dim 2')
        ax2.set_zlabel('Latent Dim 3')
        ax2.set_title(f'{ext_dataset} Centroids\nr = {overall_correlation:.4f}')

        plt.tight_layout()
        plt.show()

# ========================================================================
# SUMMARY AND CONCLUSIONS
# ========================================================================
print("="*80)
print("REPLICATION TESTING SUMMARY")
print("="*80)
print("\n⚠ METHODOLOGICAL NOTES:")
print("  - Correlation computed on centroids (n=K, typically 2-6)")
print("  - P-values should be interpreted with caution due to small sample size")
print("  - Correlation coefficient (r) is the primary metric for hypothesis testing")
print("  - Forced K matching: External datasets re-clustered with D1's K for comparison")
print("="*80)

replication_df = pd.DataFrame(replication_results)

display_cols = [
    'dataset',
    'replication_score',
    'avg_dim_correlation',
    'h1_supported',
    'h2_supported'
]

if any(replication_df['info_loss_pct'].notna()):
    display_cols.append('info_loss_pct')

print("\nResults Table:")
print(replication_df[display_cols].to_string(index=False))

h1_count = sum(replication_df['h1_supported'] == True)
h2_count = sum(replication_df['h2_supported'] == True)
total_tested = len(replication_df)

print(f"\n{'='*80}")
print("OVERALL CONCLUSION:")
print(f"{'='*80}")
print(f"Reference: D1-Swiss ({d1_latent_dim}D, K={d1_k})")
print(f"All datasets standardized to: {d1_latent_dim}D (K can differ)")
print(f"Datasets tested: {total_tested}")
print(f"H1 (Universal) supported: {h1_count}/{total_tested}")
print(f"H2 (Contextual) supported: {h2_count}/{total_tested}")

if h1_count >= 2:
    print("\n✓ H1 VALIDATED: Profiles appear to be UNIVERSAL across datasets")
    print("  → Mental health profiles are consistent across different populations")
elif h2_count >= 2:
    print("\n✓ H2 VALIDATED: Profiles appear to be CONTEXT-SPECIFIC")
    print("  → Mental health profiles vary by population/culture")
else:
    print("\n→ Mixed results: Profiles show moderate replication")
    print("  → May depend on specific dataset characteristics")

print(f"{'='*80}\n")


# In[ ]:


# =========================================================================
# OPTIONAL: Process All Datasets Automatically
# =========================================================================
# Uncomment the function call below to process all datasets through
# the autoencoder pipeline automatically
# =========================================================================

def run_all_datasets():
    """
    Process all datasets through the autoencoder pipeline.

    This function iterates through all datasets defined in DATASETS
    and runs the complete pipeline (hyperparameter tuning + profile extraction)
    for each one.
    """
    for dataset_name in DATASETS:
        print(f"\n{'#'*80}\nProcessing {dataset_name}\n{'#'*80}\n")
        run_autoencoder_pipeline(dataset_name)

# Uncomment the line below to run:
# run_all_datasets()

