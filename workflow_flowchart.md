# Mental Health Profiling Workflow - Flowchart

## Complete Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 0: DATA PREPARATION                            │
│                    (Cell 0)                                             │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  D1_Swiss_processed.csv              │
        │  Features: [Depression, Anxiety,     │
        │            Stress, Burnout]          │
        │  Shape: (N samples × 4 features)     │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  Split: 80% train+val, 20% test    │
        │  train_val_data: (N×4)              │
        │  test_data: (held out)               │
        └─────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: AUTOENCODER ARCHITECTURE                   │
│                    (Cell 1)                                             │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  Autoencoder Class                 │
        │  Encoder: 4D → hidden → 2D/3D     │
        │  Decoder: 2D/3D → hidden → 4D     │
        └─────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: HYPERPARAMETER TUNING                      │
│                    (Cells 2-6)                                          │
└─────────────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                             │
        ▼                                             ▼
┌──────────────────────┐                  ┌──────────────────────┐
│  STAGE 1:            │                  │  STAGE 2:             │
│  Architecture Tuning  │                  │  Learning Params      │
│  (Cells 3-4)          │                  │  (Cells 5-6)         │
└──────────────────────┘                  └──────────────────────┘
        │                                             │
        │  For each config:                          │  For each config:
        │  1. Train autoencoder                      │  1. Train autoencoder
        │  2. Encode → latent vectors                │  2. Encode → latent vectors
        │  3. Cluster (K=2-6)                       │  3. Cluster (K=2-6)
        │  4. Pick best K (silhouette)               │  4. Pick best K (silhouette)
        │  5. Record metrics                         │  5. Record metrics
        │                                             │
        ▼                                             ▼
┌──────────────────────┐                  ┌──────────────────────┐
│  Best Architecture:  │                  │  Best Learning Params:│
│  - hidden_size       │                  │  - learning_rate      │
│  - latent_dim        │                  │  - batch_size         │
│  - activation        │                  │  - weight_decay        │
│  - optimizer         │                  │  - momentum           │
│  - epochs            │                  │  - optimal K          │
│  - optimal K         │                  │                       │
└──────────────────────┘                  └──────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 3: FINAL MODEL TRAINING                       │
│                    (Cell 7)                                            │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  Train Final Model                  │
        │  Using best hyperparameters         │
        │  On ALL train+val data              │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  Encode All D1 Samples              │
        │  train_val_data (N×4)               │
        │           ↓                         │
        │  d1_latent_vectors (N×2 or N×3)    │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  K-means Clustering                 │
        │  Input: d1_latent_vectors           │
        │  K: optimal K (from tuning)        │
        │           ↓                         │
        │  d1_cluster_labels (N×1)           │
        │  d1_centroids (K×2 or K×3)        │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  INTERPRETATION: Map Back to        │
        │  Original Symptom Space            │
        │                                     │
        │  For each cluster k:                │
        │  1. cluster_mask = labels == k     │
        │  2. cluster_data =                 │
        │     train_val_data[cluster_mask]   │
        │     (Original 4D symptoms!)         │
        │  3. feature_means =                 │
        │     cluster_data.mean(axis=0)      │
        │     [Dep, Anx, Str, Bur]            │
        └─────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  Profile Summary                    │
        │  P1: High Dep+Anx, Low Str+Bur     │
        │  P2: Low Dep+Anx, High Str+Bur    │
        │  P3: Medium all symptoms           │
        │  ...                                │
        └─────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 4: H3 VALIDATION                              │
│                    (Cell 8)                                            │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  Load Therapy Use Data               │
        │  Create Contingency Table:          │
        │  Cluster × Therapy Use              │
        │           ↓                         │
        │  Chi-Square Test                    │
        │  H3: p < 0.05?                      │
        └─────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 5: REPLICATION TESTING                       │
│                    (Cell 9)                                            │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  For each External Dataset          │
        │  (D2, D3, D4):                      │
        │                                     │
        │  1. Load & preprocess               │
        │  2. Encode with trained model       │
        │     → ext_latent_vectors            │
        │  3. Cluster → ext_centroids         │
        │  4. Match centroids (nearest)       │
        │  5. Compute correlations            │
        │  6. Average → replication score r   │
        │  7. Classify:                       │
        │     - H1 (Universal): r > 0.70      │
        │     - H2 (Contextual): r < 0.50    │
        └─────────────────────────────────────┘
```

## Key Data Flow Diagram

```
ORIGINAL DATA SPACE (4D)          LATENT SPACE (2D/3D)          CLUSTER SPACE (1D)
─────────────────────────         ────────────────────         ───────────────────
                                                                
Sample 0:                        Sample 0:                      Sample 0:
[Dep=5.2, Anx=4.8,              [lat1=0.3, lat2=-0.7]          Cluster 2
 Str=3.1, Bur=2.9]               │                              │
        │                        │                              │
        │                        │                              │
        └────────────────────────┼──────────────────────────────┘
                                 │                              │
                                 │                              │
Sample 1:                        Sample 1:                      Sample 1:
[Dep=6.1, Anx=5.9,              [lat1=0.4, lat2=-0.6]          Cluster 2
 Str=2.8, Bur=2.5]               │                              │
        │                        │                              │
        │                        │                              │
        └────────────────────────┼──────────────────────────────┘
                                 │                              │
                                 │                              │
        ENCODER                  │         K-MEANS              │
        (4D → 2D)                │         CLUSTERING           │
                                 │                              │
                                 ▼                              ▼
                                                         
                    CLUSTER INTERPRETATION (Back to 4D)
                    ────────────────────────────────────
                    
                    Cluster 2 samples:
                    - Sample 0: [Dep=5.2, Anx=4.8, Str=3.1, Bur=2.9]
                    - Sample 1: [Dep=6.1, Anx=5.9, Str=2.8, Bur=2.5]
                    - ...
                    
                    Mean: [Dep=5.8, Anx=5.2, Str=2.9, Bur=2.6]
                    
                    → Profile 2: "High Depression+Anxiety, Low Stress+Burnout"
```

## Why We Use train_val_data for Interpretation

```
┌─────────────────────────────────────────────────────────────────┐
│  CLUSTERING (in Latent Space)                                   │
│  ────────────────────────────────                               │
│                                                                  │
│  d1_latent_vectors: (N × 2)                                     │
│  [Sample 0: [0.3, -0.7],                                       │
│   Sample 1: [0.4, -0.6],                                       │
│   Sample 2: [0.1, 0.8],                                        │
│   ...]                                                          │
│                                                                  │
│  ↓ K-means clustering                                            │
│                                                                  │
│  d1_cluster_labels: (N × 1)                                    │
│  [Sample 0: Cluster 2,                                         │
│   Sample 1: Cluster 2,                                         │
│   Sample 2: Cluster 0,                                         │
│   ...]                                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ cluster_mask = labels == 2
                              │ (tells us which samples)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  INTERPRETATION (in Original Space)                             │
│  ───────────────────────────────────                            │
│                                                                  │
│  train_val_data: (N × 4)                                        │
│  [Sample 0: [Dep=5.2, Anx=4.8, Str=3.1, Bur=2.9],             │
│   Sample 1: [Dep=6.1, Anx=5.9, Str=2.8, Bur=2.5],             │
│   Sample 2: [Dep=2.1, Anx=1.9, Str=6.8, Bur=7.5],             │
│   ...]                                                          │
│                                                                  │
│  ↓ Select samples using cluster_mask                            │
│                                                                  │
│  cluster_data = train_val_data[cluster_mask]                   │
│  [Sample 0: [Dep=5.2, Anx=4.8, Str=3.1, Bur=2.9],             │
│   Sample 1: [Dep=6.1, Anx=5.9, Str=2.8, Bur=2.5],             │
│   ...]                                                          │
│                                                                  │
│  ↓ Compute mean                                                  │
│                                                                  │
│  feature_means = [Dep=5.8, Anx=5.2, Str=2.9, Bur=2.6]         │
│                                                                  │
│  → "Cluster 2: High Depression+Anxiety, Low Stress+Burnout"    │
└─────────────────────────────────────────────────────────────────┘
```

## The Complete Picture

```
┌──────────────────────────────────────────────────────────────┐
│  STEP 1: CLUSTER IN LATENT SPACE                              │
│  ─────────────────────────────────                            │
│  Why? Latent space captures patterns/relationships           │
│  Input: d1_latent_vectors (N × 2)                           │
│  Output: d1_cluster_labels (N × 1)                           │
│  Result: "Samples 0, 1, 5, 12... are similar in latent space"│
└──────────────────────────────────────────────────────────────┘
                              │
                              │ Use labels to select samples
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  STEP 2: INTERPRET IN ORIGINAL SPACE                          │
│  ─────────────────────────────────                            │
│  Why? Original space is interpretable (Depression, Anxiety...)│
│  Input: train_val_data[cluster_mask] (M × 4)                │
│  Output: feature_means [Dep, Anx, Str, Bur]                  │
│  Result: "These samples have high Dep+Anx, low Str+Bur"      │
└──────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  PROFILE SUMMARY │
                    │  P1: High Dep+Anx│
                    │  P2: Low Dep+Anx │
                    │  P3: Medium all  │
                    └──────────────────┘
```

## Summary

1. **Clustering happens in LATENT SPACE** (2D/3D) - finds patterns
2. **Interpretation happens in ORIGINAL SPACE** (4D) - explains what patterns mean
3. **The connection**: Both have same N samples, so cluster labels map to original data
4. **Why both?**: Latent space is good for clustering, original space is interpretable

