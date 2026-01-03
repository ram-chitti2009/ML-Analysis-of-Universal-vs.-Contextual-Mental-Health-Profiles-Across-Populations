import json
import time
import warnings
from collections import defaultdict, Counter
from pathlib import Path

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
    "D1_Swiss_processed": Path("/content/D1_Swiss_processed.csv"),
    "D2_Cultural_Processed": Path("/content/D2_Cultural_processed.csv"),
    "D3_Academic_Processed": Path("/content/D3_Academic_processed.csv"),
    "D4_Tech_Processed": Path("/content/D4_Tech_processed.csv"),
}
# Optimal bin numbers for stratified splitting (from balance_check.ipynb grid search)
STRATIFICATION_BINS = {
    "D1_Swiss_processed": 2,
    "D2_Cultural_Processed": 2,
    "D3_Academic_Processed": 4,
    "D4_Tech_Processed": 2,
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

    # Handle NaN values in feature columns
    if df[FEATURE_COLUMNS].isnull().values.any():
        print(f" Warning: NaN values found in {dataset_name} feature columns. Imputing with mean.")
        df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(df[FEATURE_COLUMNS].mean())

    feature_matrix = df[FEATURE_COLUMNS].values
    print(f"Dataset: {feature_matrix.shape[0]} samples, {feature_matrix.shape[1]} features")

    # Check if we have enough samples for train/test split
    if len(feature_matrix) < 10:
        raise ValueError(f"Dataset too small ({len(feature_matrix)} samples). Need at least 10 samples.")

    # STRATIFIED SPLIT: Use binned features for stratification
    # Create bins for each continuous feature to enable stratification
    # This ensures train/test sets have similar distributions
    try:
        n_bins = STRATIFICATION_BINS.get(dataset_name, 2)

        df_binned = df.copy()
        for col in FEATURE_COLUMNS:
            df_binned[f'{col}_bin'] = pd.qcut(df[col], q=n_bins, labels=False, duplicates='drop')

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
        print(f"Using STRATIFIED split with {n_bins} bins per feature (ensures balanced train/test distributions)")

    except (ValueError, KeyError) as e:
        print(f"Stratification failed ({str(e)}), using regular split")
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

# MAIN PIPELINE - Hyperparameter Tuning and Profile Extraction
# Two-stage hyperparameter optimization followed by profile extraction
# Stage 1: Architecture tuning (hidden_size, latent_dim, activation, optimizer)
# Stage 2: Learning parameter tuning (batch_size, weight_decay, momentum)
# Final: Train on all data, extract profiles, evaluate on test set

def run_autoencoder_pipeline(dataset_name: str, force_latent_dim: int = None, force_k: int = None, narrowed_params: dict = None):
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

    # STAGE 1: Architecture Parameter Tuning
    # Grid search over architecture parameters with 10-fold cross-validation
    # Selection criterion: Consensus voting on clustering quality metrics
    print("STAGE 1: Architecture Parameters Tuning (K-Fold CV)")

    # Use narrowed params from Optuna scout if provided
    if narrowed_params:
        print("Using NARROWED parameter ranges from Optuna scout phase")
        hidden_sizes = narrowed_params.get("hidden_sizes", [2, 3, 4, 5, 6, 8, 10])
        latent_dims = narrowed_params.get("latent_dims", [2, 3, 4])

        all_activations = {"ReLU": nn.ReLU, "Tanh": nn.Tanh, "Sigmoid": nn.Sigmoid}
        activations = {name: all_activations[name] for name in narrowed_params.get("activations", ["ReLU", "Tanh", "Sigmoid"])}

        all_optimizers = {"Adam": optim.Adam, "SGD": optim.SGD}
        optimizers = {name: all_optimizers[name] for name in narrowed_params.get("optimizers", ["Adam", "SGD"])}

        epochs_list = narrowed_params.get("epochs", [50, 75, 100])
    else:
        print("Using FULL parameter ranges (no scout phase)")

        hidden_sizes = [2, 3, 4, 5, 6, 8, 10]

        # Latent dimension grid: [2, 3, 4]
        # Justification:
        # - 2: Standard for 2D visualization and clustering (most interpretable)
        # - 3: Allows 3D visualization, more capacity for complex manifolds
        # - 4: Maximum reasonable for 4-feature input (no compression if latent_dim >= input_dim)
        # - Excludes 1: 1D clustering is degenerate (all points on a line, poor separation)
        latent_dims = [2, 3, 4]

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
    print(f"Total: {total_experiments} experiments ({total_experiments//n_folds} configs \u00d7 {n_folds} folds)\n")

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

    print(f"\ Stage 1 completed in {(time.time()-start_stage1)/60:.2f} minutes\n")

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
    print(f"  Silhouette: {best_config_stage1['mean_silhouette']:.6f} \u00b1 {best_config_stage1['std_silhouette']:.6f}")
    print(f"  Calinski-Harabasz: {best_config_stage1['mean_ch']:.6f} \u00b1 {best_config_stage1['std_ch']:.6f}")
    print(f"  Davies-Bouldin: {best_config_stage1['mean_db']:.6f} \u00b1 {best_config_stage1['std_db']:.6f}")
    print(f"  Reconstruction loss: {best_config_stage1['mean_recon_loss']:.6f} \u00b1 {best_config_stage1['std_recon_loss']:.6f}")
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
    train_loader_lr = DataLoader(train_subset, batch_size=16, shuffle=True) # Changed batch_size to 16 from 64 so that the iterations are matched.
    criterion = nn.MSELoss()

    print(f"\nLR Range Test: {best_optimizer_name} optimizer")
    model_lr = Autoencoder(INPUT_DIM, best_hidden_size, best_latent_dim, best_activation_fn).to(device)

    if best_optimizer_name == 'Adam':
        optimizer_lr = optim.Adam(model_lr.parameters(), lr=1e-7)
    else:
        optimizer_lr = optim.SGD(model_lr.parameters(), lr=1e-7, momentum=0.9)

    lr_finder = LRFinder(model_lr, optimizer_lr, criterion, device=device)
    lr_finder.range_test(train_loader_lr, end_lr=10, num_iter=100, step_mode="exp")
    lr_finder.plot() # This is where the error occurred, needs sufficient iterations generated by the loader
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
    print(f"Total: {total_experiments} experiments ({total_experiments//n_folds} configs \u00d7 {n_folds} folds)\n")

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

    best_learning_rate = float(best_config_stage2['learning_rate'])
    best_batch_size = int(best_config_stage2['batch_size'])
    best_weight_decay = float(best_config_stage2['weight_decay'])
    best_momentum = float(best_config_stage2.get('momentum', 0.0)) if 'momentum' in best_config_stage2 else None

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
    print(f"  Silhouette: {best_config_stage2['mean_silhouette']:.6f} \u00b1 {best_config_stage2['std_silhouette']:.6f}")
    print(f"  Calinski-Harabasz: {best_config_stage2['mean_ch']:.6f} \u00b1 {best_config_stage2['std_ch']:.6f}")
    print(f"  Davies-Bouldin: {best_config_stage2['mean_db']:.6f} \u00b1 {best_config_stage2['std_db']:.6f}")
    print(f"  Reconstruction loss: {best_config_stage2['mean_recon_loss']:.6f} \u00b1 {best_config_stage2['std_recon_loss']:.6f}")
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
        print(f"\u26a0\ufe0f ERROR: Final clustering produced only {n_unique_clusters} cluster(s). Cannot compute metrics.")
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
        print(f"  train_val_data[0, {i}] = {sample_0[i]:.4f} \u2192 {feature_name}")
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
        print(f"\u26a0\ufe0f ERROR: Test clustering produced only {n_unique_test_clusters} cluster(s). Cannot compute metrics.")
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
        print(f"\n\u2713 Good generalization: Test performance within {sil_diff_pct:.2f}% of train+val")
    elif sil_diff_pct < 10:
        print(f"\n\u26a0\ufe0f Moderate generalization gap: Test performance {sil_diff_pct:.2f}% different from train+val")
    else:
        print(f"\n\u2718 Potential overfitting: Test performance {sil_diff_pct:.2f}% different from train+val")

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
        print(f"\n\u2713 Data aligned: {len(all_cluster_labels)} samples")
        print(f"  Therapy use rate: {y_therapy_aligned.mean():.2%} ({y_therapy_aligned.sum()}/{len(y_therapy_aligned)})")

        print("\nChi-Square Test for Independence:")
        print("="*70)

        contingency = pd.crosstab(all_cluster_labels, y_therapy_aligned)
        chi2, p, dof, expected = chi2_contingency(contingency)

        print("   Contingency Table:")
        print(contingency)
        print(f"\n   Chi-square statistic: \u03c7\u00b2 = {chi2:.4f}")
        print(f"   Degrees of freedom: df = {dof}")
        print(f"   p-value: p = {p:.6f}")

        alpha = 0.05
        if p < alpha:
            print(f"\n   \u2713 SIGNIFICANT (p < {alpha}): Profile membership IS associated with therapy utilization")
            print("   \u2192 H3 VALIDATED: Profiles have clinical utility")
        else:
            print(f"\n   \u2718 NOT SIGNIFICANT (p >= {alpha}): No association detected")
            print("   \u2192 H3 NOT VALIDATED")

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
                    print(f"      \u2022 {profile_name} - {therapy_status}: {direction} than expected (residual = {residuals[i, j]:.2f})")

        print("\n" + "="*70)
        print("H3 VALIDATION SUMMARY:")
        print("="*70)
        print(f"Dataset: {dataset_name} (N={len(all_cluster_labels)})")
        print(f"Chi-square: \u03c7\u00b2 = {chi2:.4f}, p = {p:.6f}, df = {dof}")
        print(f"Cram\u00e9r's V = {cramers_v:.4f} ({effect_size} effect)")
        print(f"H3 Status: {' \u2713 VALIDATED' if p < alpha else '\u2718 NOT VALIDATED'}")

        if p < alpha:
            print("\nConclusion:")
            print("  Profiles demonstrate clinical utility by predicting therapy utilization.")
            print("  This supports the use of these profiles for targeted mental health interventions.")

        print("="*70 + "\n")
    else:
        print("\nSkipping H3 validation (only available for D1-Swiss with PSYT_Therapy_Use)")

    # Prepare results for saving
    overall_results = {
        'dataset_name': dataset_name,
        'best_architecture_params': {
            'hidden_size': best_hidden_size,
            'latent_dim': best_latent_dim,
            'activation': best_activation_name,
            'optimizer': best_optimizer_name,
            'epochs': best_epochs
        },
        'best_learning_params': {
            'learning_rate': best_learning_rate,
            'batch_size': best_batch_size,
            'weight_decay': best_weight_decay,
            'momentum': best_momentum
        },
        'final_model_performance': {
            'optimal_k': best_k,
            'train_val_silhouette_score': final_sil_score,
            'train_val_calinski_harabasz_score': final_ch_score,
            'train_val_davies_bouldin_score': final_db_score,
            'test_reconstruction_loss_mse': test_recon_loss,
            'test_silhouette_score': test_sil_score,
            'test_calinski_harabasz_score': test_ch_score,
            'test_davies_bouldin_score': test_db_score
        },
        'profile_summary': profile_summary, # This is already a list of dicts
        'interpreted_profiles_df': interpreted_df.to_dict('records') # Convert DataFrame to list of dicts
    }

    # Define save path
    output_filename = f"{dataset_name}_autoencoder_results.json"
    output_path = Path.cwd() / output_filename # Current working directory is /content/drive/MyDrive/CAPTURE

    # Save results to JSON
    try:
        with open(output_path, 'w') as f:
            json.dump(overall_results, f, indent=4)
        print(f"\n\u2713 Successfully saved pipeline results to {output_path}")
    except Exception as e:
        print(f"\u2718 Error saving pipeline results: {e}")

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