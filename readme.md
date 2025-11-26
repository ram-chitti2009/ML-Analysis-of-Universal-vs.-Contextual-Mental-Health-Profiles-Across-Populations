CAPTURE - Cross Dataset Analysis using Unsupervised REsearch

##Overview
Mental Health profiling system using Autonencoder and clustering to identify to identify different profiles based on metrics such as Anxiety, Depression, Burnout, and Stress

##Quick Start
### Prerequisites
- Python 3.x
- PyTorch
- Scikit-learn
- See requirements.txt 

### Installation
```bash
pip install -r requirements.txt
```

### Usage
1. Place datasets in the project directory:
   - `D1_Swiss_processed.csv`
   - `D2_Cultural_processed.csv`
   - `D3_Academic_processed.csv`
   - `D4_Tech_processed.csv`

2. Run `Autoencoder.ipynb` cells in order:
   - Cell 0: Setup and data preparation
   - Cell 1-7: Hyperparameter tuning and profile extraction
   - Cell 5: Replication testing (H1/H2)

### Expected Runtime
- Per dataset: 15-45 minutes (CPU) or 5-15 minutes (GPU)
- All 4 datasets: 1-3 hours (CPU) or 20-60 minutes (GPU)

## Methodology
See `METHODOLOGY.md` for complete methodology documentation.
