#Optuna Scout Phase-
#The purpose of this phase is to intelligently explore the hyperparameter space
#Uses 3fold cv, 15 epochs(expedited)

import optuna
from optuna.pruners import MedianPruner

# train_and_validate_model is defined in Cell 3, no import needed

def scout_objective(trial, dataset_name="D1_Swiss_Processed"):
    """
    Objective function for Scout hyperparameter optimization.
    """
    #Suggest Hyperparameters
    hidden_size = trial.suggest_int('hidden_size', 2, 12)  # Include 2 for 4-feature datasets
    latent_dim = trial.suggest_int('latent_dim', 2, 4)
    activation = trial.suggest_categorical('activation', ['ReLU', 'Tanh', 'Sigmoid'])
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])



    (df, feature_matrix, train_val_data, test_data,
    train_val_tensor, test_tensor, kfold, dataset_path) = prepare_dataset(
        dataset_name
    )

    kfold_scout = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

    activation_map = {
        'ReLU': nn.ReLU,
        'Tanh': nn.Tanh,
        'Sigmoid': nn.Sigmoid
    }

    scores = []
    criterion = nn.MSELoss()
    fixed_lr = 1e-3
    for fold_idx, (train_idx, val_idx) in enumerate(kfold_scout.split(train_val_tensor)):
        fold_seed = RANDOM_SEED + fold_idx
        torch.manual_seed(fold_seed)
        np.random.seed(fold_seed)


        train_fold = train_val_tensor[train_idx]
        val_fold = train_val_tensor[val_idx]

        train_dataset = TensorDataset(train_fold.cpu())
        val_dataset = TensorDataset(val_fold.cpu())
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        model = Autoencoder(
            len(FEATURE_COLUMNS),
            hidden_size,
            latent_dim,
            activation_map[activation]

        ).to(device)

        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=fixed_lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=fixed_lr, momentum=0.9)

        # Train and Validate - 15 epochs only
        scout_epochs = 15

        train_losses, val_losses, optimal_k, best_sil_score, latent_vectors, validation_metrics = \
            train_and_validate_model(model, train_loader, val_loader, optimizer,
                                    criterion, scout_epochs, device)

        fold_score = best_sil_score
        scores.append(fold_score)

        # Report intermediate value (enables per-fold pruning)
        # Each fold = 1 step, so with 3-fold CV: 5 steps = 5 ÷ 3 = 1.67 folds
        trial.report(fold_score, step=fold_idx)

        # Check if should prune after this fold
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores) if scores else 0.0
def run_scout_phase(dataset_name="D1_Swiss_Processed", n_trials=150, save_path=None):
    print("="*80)
    print(f"SCOUT PHASE: Optuna Optimization on {dataset_name}")
    print("="*80)
    print(f"Configuration:")
    print(f"  - Trials: {n_trials}")
    print(f"  - CV Folds: 3 (fast)")
    print(f"  - Epochs: 15 (fast)")
    print(f"  - Pruning: Enabled (stops bad trials early)")
    print("="*80)

    study = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner(
            n_startup_trials=20,  # Wait for 20 complete trials before pruning
                              # (Ensures stable median baseline, prevents false positives)
            n_warmup_steps=5      # Wait for 5 reporting steps before pruning
                              # With 3-fold CV: 5 steps ÷ 3 folds = 1.67 folds
                              # (Allows 1 complete fold + 2/3 of next fold before pruning)
        )
    )
    start_time = time.time()

    study.optimize(
        lambda trial: scout_objective(trial, dataset_name),
        n_trials=n_trials,
        show_progress_bar=True
    )

    elapsed_time = time.time() - start_time

    print(f"\nScout phase completed in {elapsed_time/60:.1f} minutes")
    print(f"  Best trial: {study.best_trial.number}")
    print(f"  Best value: {study.best_value:.6f}")
    print(f"  Best params: {study.best_trial.params}")

    top_5_trials = sorted([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE], key=lambda x: x.value if x.value is not None else -1, reverse=True)[:5]

    print(f"\nTop 5 Trials:")
    for i, trial in enumerate(top_5_trials, 1):
        print(f"  {i}. Trial {trial.number}: Score={trial.value:.6f}, Params={trial.params}")

    # Analyze patterns
    print(f"\nPattern Analysis:")
    hidden_sizes = [t.params['hidden_size'] for t in top_5_trials]
    latent_dims = [t.params['latent_dim'] for t in top_5_trials]
    activations = [t.params['activation'] for t in top_5_trials]
    optimizers = [t.params['optimizer'] for t in top_5_trials]

    print(f"  Hidden sizes in top 5: {Counter(hidden_sizes)}")
    print(f"  Latent dims in top 5: {Counter(latent_dims)}")
    print(f"  Activations in top 5: {Counter(activations)}")
    print(f"  Optimizers in top 5: {Counter(optimizers)}")

    # Save results
    scout_results = {
        'dataset': dataset_name,
        'n_trials': n_trials,
        'best_trial': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_trial.params,
        'top_5_trials': [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params
            } for t in top_5_trials
        ],
        'patterns': {
            'hidden_sizes': dict(Counter(hidden_sizes)),
            'latent_dims': dict(Counter(latent_dims)),
            'activations': dict(Counter(activations)),
            'optimizers': dict(Counter(optimizers))
        }
    }

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(scout_results, f, indent=2)
        print(f"\nResults saved to {save_path}")

    return study, scout_results
