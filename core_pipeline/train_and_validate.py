# TRAINING AND VALIDATION FUNCTION
# Train autoencoder and evaluate clustering quality in latent space
# Key feature: Model quality evaluated by clustering metrics, not just
#              reconstruction loss, because goal is finding distinct profiles

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
        epoch_training_loss = 0.0
        for batch_data in train_loader:
            batch_data = batch_data[0].to(device)
            optimizer.zero_grad()
            reconstructed = model(batch_data)
            loss = criterion(reconstructed, batch_data)
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

    # EXTRACT LATENT VECTORS AND EVALUATE CLUSTERING QUALITY
    # After training, extract latent representations and evaluate clustering
    model.eval()
    all_latent_vectors = []

    with torch.no_grad():
        for batch_data in train_loader:
            data = batch_data[0].to(device)
            latent = model.encoder(data)
            all_latent_vectors.append(latent.cpu().numpy())

        for batch_data in val_loader:
            data = batch_data[0].to(device)
            latent = model.encoder(data)
            all_latent_vectors.append(latent.cpu().numpy())

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
        print("WARNING: MODEL COLLAPSED - All latent vectors are nearly identical!")
        print("   This means the encoder isn't learning useful representations.")
    print("="*70)

    # K Selection via Multiple Validation Methods (Convergent Validity)
    k_range = range(2, 7)

    silhouette_scores_list = []
    calinski_harabasz_scores = []
    davies_bouldin_scores = []
    wcss_values = []

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
            sil_score = -1
            ch_score = 0
            db_score = float('inf')
        else:
            print("Computing metrics...", end=" ", flush=True)
            sil_score = silhouette_score(latent_vectors, cluster_labels)
            ch_score = calinski_harabasz_score(latent_vectors, cluster_labels)
            db_score = davies_bouldin_score(latent_vectors, cluster_labels)
            print(f"Sil={sil_score:.4f}, CH={ch_score:.2f}, DB={db_score:.4f}")
        wcss = kmeans.inertia_

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

    # Consensus/Voting Approach for K Selection
    # Determine optimal K from each method
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores_list)]
    optimal_k_ch = k_range[np.argmax(calinski_harabasz_scores)]
    optimal_k_db = k_range[np.argmin(davies_bouldin_scores)]

    # Elbow Method: Find knee where WCSS decrease rate slows
    wcss_decreases = []
    for i in range(1, len(wcss_values)):
        if wcss_values[i-1] > 0:
            pct_decrease = ((wcss_values[i-1] - wcss_values[i]) / wcss_values[i-1]) * 100
            wcss_decreases.append(pct_decrease)
        else:
            wcss_decreases.append(0)

    if len(wcss_decreases) > 0:
        decrease_rates = np.array(wcss_decreases)
        if len(decrease_rates) > 1:
            rate_changes = np.diff(decrease_rates)
            elbow_idx = np.argmin(rate_changes) + 1
            elbow_idx = min(elbow_idx, len(k_range) - 1)
            optimal_k_elbow = k_range[elbow_idx]
        else:
            min_decrease_idx = np.argmin(wcss_decreases) + 1
            min_decrease_idx = min(min_decrease_idx, len(k_range) - 1)
            optimal_k_elbow = k_range[min_decrease_idx]
    else:
        optimal_k_elbow = optimal_k_silhouette

    # Consensus voting: If 2+ methods agree, use that K

    k_votes = [optimal_k_silhouette, optimal_k_ch, optimal_k_db, optimal_k_elbow]
    k_counts = Counter(k_votes)
    most_common_k, consensus_count = k_counts.most_common(1)[0]

    if consensus_count >= 2:
        optimal_k = most_common_k
        consensus_status = f"Consensus: {consensus_count} methods agree on K={optimal_k}"
        consensus_reached = True
    else:
        optimal_k = optimal_k_silhouette
        consensus_status = f"No consensus (Sil={optimal_k_silhouette}, CH={optimal_k_ch}, DB={optimal_k_db}, Elbow={optimal_k_elbow}). Using silhouette K={optimal_k}"
        consensus_reached = False

    print(f"Optimal K selected: {optimal_k} ({consensus_status})")
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