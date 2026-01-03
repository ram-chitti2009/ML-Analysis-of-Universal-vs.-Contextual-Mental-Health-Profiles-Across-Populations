#Extract the narrowed ranges from the optuna exploration study
#Analyse the patterns and trends in the results
#Use these ranges to set the hyperparameters for the next phase

def extract_narrowed_ranges(scout_results):
    """
    Extract narrowed ranges from scout results
    """
    top_5 = scout_results['top_5_trials']

    #Extract ranges from top 5 trials
    hidden_sizes = sorted(set([t['params']['hidden_size'] for t in top_5]))
    latent_dims = sorted(set([t['params']['latent_dim'] for t in top_5]))
    activations = sorted(set([t['params']['activation'] for t in top_5]))
    optimizers = sorted(set([t['params']['optimizer'] for t in top_5]))


    hidden_sizes_expanded = sorted(set(
        [max(3, h-1) for h in hidden_sizes] +
        hidden_sizes +
        [min(12, h+1) for h in hidden_sizes]
    ))

    if len(activations) == 3:
        act_counts = Counter([t['params']['activation'] for t in top_5])
        activations = [act for act in activations if act_counts[act] >= 2]

    narrowed_params = {
        'hidden_sizes': hidden_sizes_expanded,
        'latent_dims': latent_dims,
        'activations': activations,
        'optimizers': optimizers,
        'epochs': [50, 75, 100]  # Full epochs for focused grid
    }

    print("Narrowed Parameter Ranges for Focused Grid:")
    print(f"  hidden_sizes: {narrowed_params['hidden_sizes']}")
    print(f"  latent_dims: {narrowed_params['latent_dims']}")
    print(f"  activations: {narrowed_params['activations']}")
    print(f"  optimizers: {narrowed_params['optimizers']}")
    print(f"  epochs: {narrowed_params['epochs']}")

    grid_size = (len(narrowed_params['hidden_sizes']) *
                 len(narrowed_params['latent_dims']) *
                 len(narrowed_params['activations']) *
                 len(narrowed_params['optimizers']) *
                 len(narrowed_params['epochs']))
    print(f"\n  Grid size: {grid_size} configs × 10 folds = {grid_size * 10} experiments")

    return narrowed_params

