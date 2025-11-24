# Autoencoder.ipynb - Complete Explanation

## Overview
This notebook implements a **mental health profiling system** using autoencoders and clustering to identify distinct patient profiles based on Depression, Anxiety, Stress, and Burnout symptoms.

---

## **Cell 0: Setup & Data Preparation**

**What it does:**
- Sets up reproducibility (random seeds, deterministic CUDA)
- Loads D1_Swiss_processed.csv with 4 features: [Depression, Anxiety, Stress, Burnout]
- Splits data: 80% train+val (for hyperparameter tuning), 20% test (held out)
- Initializes 10-fold cross-validation splitter

**Key outputs:**
- `train_val_data`: 80% of data for training/validation
- `test_data`: 20% held out (not used in this notebook)
- `kfold`: 10-fold CV splitter for robust hyperparameter tuning

---

## **Cell 1: Autoencoder Architecture Definition**

**What it does:**
- Defines a feedforward autoencoder neural network
- **Encoder**: Compresses 4D symptoms → hidden layer → 2D/3D latent space
- **Decoder**: Reconstructs latent space → hidden layer → 4D symptoms
- Uses symmetric architecture (encoder/decoder mirror each other)

**Purpose:**
- Learn compressed representations that capture patterns in symptom relationships
- Latent space will be used for clustering to find patient profiles

---

## **Cell 2: Training & Validation Function**

**What it does:**
- `train_and_validate_model()` function that:
  1. **Trains** autoencoder with MSE reconstruction loss
  2. **Extracts** latent vectors from encoder (train+val combined)
  3. **Tests** K-means clustering with K=2,3,4,5,6
  4. **Evaluates** clustering quality using 3 metrics:
     - Silhouette score (primary)
     - Calinski-Harabasz index
     - Davies-Bouldin index
  5. **Selects** optimal K using consensus voting:
     - If 2+ metrics agree → use that K
     - Otherwise → use silhouette score's K

**Returns:**
- Training/validation losses
- Optimal K (number of clusters)
- Best silhouette score
- Latent vectors
- All validation metrics

**Key insight:** This function evaluates how well the autoencoder's latent space can be clustered, which determines profile quality.

---

## **Cell 3: Stage 1 - Architecture Parameter Tuning**

**What it does:**
- **Grid search** over architecture hyperparameters:
  - `hidden_size`: [3, 4, 5, 6, 8, 10]
  - `latent_dim`: [2, 3]
  - `activation`: [ReLU, Tanh, Sigmoid]
  - `optimizer`: [Adam, SGD]
  - `epochs`: [20, 50, 100]
- **Fixed**: learning_rate = 1e-3 (to isolate architecture effects)

**Process:**
- For each config × 10-fold CV:
  1. Train autoencoder on train fold
  2. Extract latent vectors (train+val combined)
  3. Test clustering with K=2-6, pick best K
  4. Record: optimal K, silhouette score, reconstruction loss

**Total experiments:** 6 × 2 × 3 × 2 × 3 × 10 = **2,160 experiments**

**Output:** `results_stage1` dictionary with all results

---

## **Cell 4: Stage 1 - Results Aggregation**

**What it does:**
- Aggregates results across 10 folds for each config
- Computes mean ± std for:
  - Silhouette score (PRIMARY metric)
  - Calinski-Harabasz score
  - Davies-Bouldin score
  - Reconstruction loss
- **Consensus voting** to select best architecture:
  - Each metric votes for its top config
  - If 2+ metrics agree → use that config
  - Otherwise → use silhouette score's top choice

**Output:**
- `best_hidden_size`, `best_latent_dim`, `best_activation_name`, `best_optimizer_name`, `best_epochs`, `best_k`

**Purpose:** Find the best architecture that produces the most clusterable latent space.

---

## **Cell 5: Stage 2 - Learning Parameter Optimization**

**What it does:**
- Uses best architecture from Stage 1
- **LR Range Test**: Finds optimal learning rate using LRFinder
- **Grid search** over learning parameters:
  - `batch_size`: [32, 64, 128]
  - `weight_decay`: [0, 1e-4, 1e-3]
  - `momentum`: [0.5, 0.9, 0.95] (if SGD)
- **Fixed**: optimal LR from range test, best architecture from Stage 1

**Process:**
- Same as Stage 1: train → extract latent → cluster → evaluate
- 10-fold CV for each config

**Total experiments:** 3 × 3 × (1 or 3) × 10 = **90-270 experiments**

**Output:** `results_stage2` dictionary with all results

---

## **Cell 6: Stage 2 - Results Aggregation**

**What it does:**
- Same consensus voting approach as Stage 1
- Aggregates results across folds
- Selects best learning parameters

**Output:**
- `best_learning_rate`, `best_batch_size`, `best_weight_decay`, `best_momentum`

**Final result:** Complete best hyperparameter set (architecture + learning params)

---

## **Cell 7: Final Model Training & Profile Extraction**

**What it does:**

### Part 1: Train Final Model
- Trains final autoencoder on **ALL** train+val data (no splitting)
- Uses best hyperparameters from Stage 1 & 2
- Trains for `best_epochs` epochs

### Part 2: Extract Latent Representations
- Encodes all D1 train+val samples → `d1_latent_vectors` (N × 2 or N × 3)

### Part 3: K-means Clustering
- Clusters latent vectors with optimal K (from Stage 1)
- Gets `d1_cluster_labels` (which cluster each sample belongs to)
- Gets `d1_centroids` (cluster centers in latent space)

### Part 4: Profile Interpretation (THE KEY PART!)
- For each cluster k:
  1. `cluster_mask = d1_cluster_labels == k` (find samples in cluster k)
  2. `cluster_data = train_val_data[cluster_mask]` (get their ORIGINAL 4D symptoms)
  3. Compute mean symptoms: `feature_means = cluster_data.mean(axis=0)`
  4. Create profile summary: "Cluster k has mean [Dep, Anx, Str, Bur] = [...]"

**Why this works:**
- Clustering happens in latent space (finds patterns)
- Interpretation happens in original space (explains what patterns mean)
- Cluster labels are "pointers" that map back to original data

**Output:**
- `d1_latent_vectors`: Latent representations
- `d1_cluster_labels`: Cluster assignments
- `d1_centroids`: Cluster centers (for replication matching)
- `profile_df`: Profile summary table showing mean symptoms per cluster

**Example output:**
```
Profile  N   Depression  Anxiety  Stress  Burnout
P1       150  5.8        5.2      2.9     2.6    (High Dep+Anx, Low Str+Bur)
P2       200  2.1        1.9      6.8     7.5    (Low Dep+Anx, High Str+Bur)
P3       250  4.0        3.8      4.2     4.1    (Medium all)
```

---

## **Cell 8 & 9: Empty (Not Yet Implemented)**

These cells should contain:
- **Cell 8**: H3 Validation (Chi-square test with therapy use)
- **Cell 9**: Replication Testing (H1/H2 on external datasets)

---

## **Overall Workflow Summary**

```
1. Setup & Load Data (Cell 0)
   ↓
2. Define Architecture (Cell 1)
   ↓
3. Create Training Function (Cell 2)
   ↓
4. Stage 1: Tune Architecture (Cells 3-4)
   - Test 2,160 configs
   - Select best architecture
   ↓
5. Stage 2: Tune Learning Params (Cells 5-6)
   - Test 90-270 configs
   - Select best learning params
   ↓
6. Train Final Model & Extract Profiles (Cell 7)
   - Train on all data
   - Cluster in latent space
   - Interpret in original space
   ↓
7. H3 Validation (Cell 8 - TODO)
   ↓
8. Replication Testing (Cell 9 - TODO)
```

---

## **Key Concepts**

### **Why Autoencoder?**
- Learns compressed representations that capture symptom relationships
- Latent space groups similar patterns together
- Better than clustering raw symptoms directly

### **Why Two-Stage Tuning?**
- Stage 1: Find best architecture (structure matters most)
- Stage 2: Fine-tune learning (optimization matters less)
- Prevents circular dependency (can't tune LR before knowing architecture)

### **Why Consensus Voting?**
- Multiple metrics provide convergent validity
- Reduces overfitting to single metric
- More robust selection

### **Why Cluster in Latent Space but Interpret in Original Space?**
- **Latent space**: Good for finding patterns (compressed, captures relationships)
- **Original space**: Good for interpretation (actual symptom values)
- Cluster labels connect the two: they're "pointers" that map back

---

## **Current Status**

✅ **Completed:**
- Setup & data preparation
- Architecture definition
- Two-stage hyperparameter tuning
- Final model training
- Profile extraction with verification

❌ **Remaining:**
- H3 validation (Chi-square test)
- Replication testing (H1/H2)

---

## **What Makes This Notebook Special**

1. **Rigorous hyperparameter tuning**: 2,000+ experiments with 10-fold CV
2. **Multi-metric evaluation**: Silhouette + CH + DB with consensus voting
3. **Automatic K selection**: Tests K=2-6, picks best per fold
4. **Verification system**: Checks that feature mapping is correct
5. **Safe interpretation**: Uses explicit dictionary mapping instead of index guessing

