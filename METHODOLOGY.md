# Methodology : CAPTURE

## 1. Introduction and Research Objectives
This document outlines the methodology used in the CAPTURE project, a cross-dataset analysis of mental health profiles using unsupervised learning. The primary goal is to identify distinct mental health profiles based on Depression, Anxiety, Stress, and Burnout symptoms, and assess their generalizability across different populations.

Hypotheses:
- **H1 (Universal Profiles)**: Mental health profiles are universal across populations (replication score r > 0.70)
- **H2 (Context-Specific Profiles)**: Mental health profiles are context-specific and vary by population/culture (replication score r < 0.50)
- **H3 (Clinical Utility)**: Profile membership is associated with therapy utilization (Chi-square test, p < 0.05)

## 2. Data Preparation

### 2.1 Datasets
Four datasets are used:
- Swiss Students (D1-Swiss) - https://www.kaggle.com/datasets/thedevastator/medical-student-mental-health
- Cultural Fragmentation (D2-Cultural) - https://www.kaggle.com/datasets/shariful07/student-mental-health
- Academic Stress (D3-Academic) - https://www.kaggle.com/datasets/adilshamim8/student-depression-dataset
- Mental Health in Tech (D4-Tech) - https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey

### 2.2 Features
All datasets contain four standardized features as part of StandardScaler:
- Depression
- Anxiety
- Stress
- Burnout

### 2.3 Data Splitting
- **Train+Validation (80%)**: Used for hyperparameter tuning and model training
- **Test (20%)**: Held out for final evaluation only
- **Random seed**: 42 (for reproducibility)

### 2.4 Cross Validation
- **10-fold Cross Validation**: Used to ensure statistically sound and rigorous evaluation of model performance

## 3. Autoencoder Architecture
- Ran PCA vs Autoencoder to compare the performance of the two models.
- **Feedforward Autoencoder**: A simple neural network with an encoder and decoder
- **Encoder**: Compresses 4D symptoms → hidden layer → 2D/3D latent space
- **Decoder**: Reconstructs latent space → hidden layer → 4D symptoms
- **MSE Loss**: Used to train the autoencoder

## 4. Hyperparameter Tuning
- **Grid Search**: Used to find the best hyperparameters for the autoencoder
- First stage is for architecture tuning and second stage is for learning parameter tuning. In between, the LR Range Test is used to find the best learning rate for the autoencoder depending on the optimizer and architecture.

### 4.1 Stage 1: Architecture Parameters
- **hidden_size**: [3, 4, 5, 6, 8, 10]
- **latent_dim**: [2, 3]
- **activation**: [ReLU, Tanh, Sigmoid]
- **optimizer**: [Adam, SGD]
- **epochs**: [20, 50, 100]
- **Fixed learning_rate**: 1e-3 (to isolate architecture effects)
- **Total experiments**: ~2,160 per dataset (6 × 2 × 3 × 2 × 3 × 10 folds)

### 4.2 Stage 2: Learning Parameters
- **learning_rate**: Determined via LR Range Test (optimal value from range test)
- **batch_size**: [32, 64, 128]
- **weight_decay**: [0, 1e-4, 1e-3]
- **momentum**: [0.5, 0.9, 0.95] (only if SGD optimizer)
- **Total experiments**: ~90-270 per dataset (3 × 3 × 1-3 × 10 folds)

## 5. Optimal K selection
- **K-means Clustering**: Used to cluster the latent vectors into different profiles
- **K range tested**: 2, 3, 4, 5, 6
- **Silhouette Score**: Used to evaluate the quality of the clusters (maximize)
- **Calinski-Harabasz Index**: Used to evaluate the quality of the clusters (maximize)
- **Davies-Bouldin Index**: Used to evaluate the quality of the clusters (minimize)
- **Elbow Method**: Used to find the optimal number of clusters (WCSS-based knee detection)
- **Consensus Voting**: Used to select the optimal number of clusters based on the majority vote of the above metrics. If 2+ methods agree on K, that K is selected. Otherwise, falls back to Silhouette score's K.

## 6. Profile Extraction
- **Profile Extraction**: Used to extract the profiles from the latent vectors by clustering in latent space and interpreting in original symptom space

## 7. Replication Testing
- **Pearson Correlation**: Used to measure the similarity between the centroids of the reference dataset (D1-Swiss) and the other datasets
- **Linear Sum Assignment Problem**: Used to match the clusters of the reference dataset to the clusters of the other datasets (Hungarian Algorithm)
- **Replication Score**: Correlation coefficient (r) used to measure profile similarity across datasets
- **H1 Supported**: r > 0.70 indicates universal profiles across populations
- **H2 Supported**: r < 0.50 indicates context-specific profiles that vary by population

## 8. Clinical Utility Testing
- **Chi-square Test**: Used to test the association between the profile membership and the therapy utilization
- **Cramer's V**: Used to measure the strength of the association between the profile membership and the therapy utilization
- **Post-Hoc Analysis**: Used to analyze the residuals of the chi-square test

## 9. Statistical Testing
Correlation test is used to check the replication score of the profiles across the datasets. P-values are interpreted with caution due to small sample size (n=K, typically 2-6 clusters).




