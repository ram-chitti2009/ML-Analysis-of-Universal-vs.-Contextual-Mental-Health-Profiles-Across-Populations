import numpy as np
import pandas as pd
import sys
import joblib
import re
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DATA_DIR = Path("./data")

# Fixed file paths to match actual filenames in data directory
FILE_PATHS = {
    "D1-Swiss": "D1-Swiss.csv",
    "D2-Cultural": "D2-Cultural.csv",
    "D3-Academic": "D3-Academic.csv",
    "D4-Tech": "D4-Tech.csv",
}

OUTPUT_PATHS = {
    "D1-Swiss": "D1_Swiss_processed.csv",
    "D2-Cultural": "D2_Cultural_processed.csv",
    "D3-Academic": "D3_Academic_processed.csv",
    "D4-Tech": "D4_Tech_processed.csv",
}

COLUMN_MAPPING = {
    "D1-Swiss": {
        "cesd": "Depression",
        "stai_t": "Anxiety",
        "mbi_ex": "Burnout",
        "mbi_cy": "Stress",
        "psyt": "PSYT_Therapy_Use",
    },
    "D2-Cultural": {
        "Do you have Depression?": "Depression",
        "Do you have Anxiety?": "Anxiety",
        "Do you have Panic attack?": "Burnout",
        "Your current year of Study": "Stress",
    },
    "D3-Academic": {
        "Depression": "Depression",
        "Academic Pressure": "Anxiety",
        "Study Satisfaction": "Burnout",
        "Financial Stress": "Stress",
    },
    "D4-Tech": {
        "mental_health_consequence": "Depression",
        "work_interfere": "Anxiety",
        "leave": "Burnout",
        "Age": "Stress",
        "treatment": "H3_Tech_Validation",
    },
}

UNIVERSAL_FEATURES = ["Depression", "Anxiety", "Burnout", "Stress"]

# Process each dataset separately with individual normalization
processed_datasets = {}
missing_sources = []

for source_name, file_name in FILE_PATHS.items():
    file_path = DATA_DIR / file_name

    if not file_path.exists():
        print(f"File not found for {source_name}: {file_path}")
        missing_sources.append(source_name)
        continue

    try:
        print(f"\n{'='*60}")
        print(f"Processing {source_name}")
        print(f"{'='*60}")
        
        df_raw = pd.read_csv(file_path, encoding="utf-8", on_bad_lines="skip")
        print(f"Loaded {source_name} dataset with shape: {df_raw.shape}")

        current_mapping = COLUMN_MAPPING[source_name]
        missing_cols = [src for src in current_mapping if src not in df_raw.columns]
        if missing_cols:
            print(f"ERROR: Missing crucial columns in {file_name}: {missing_cols}")
            missing_sources.append(source_name)
            continue

        df_named = df_raw.rename(columns=current_mapping)

        selected_columns = UNIVERSAL_FEATURES.copy()
        if "PSYT_Therapy_Use" in current_mapping.values():
            selected_columns.append("PSYT_Therapy_Use")
        if "H3_Tech_Validation" in current_mapping.values():
            selected_columns.append("H3_Tech_Validation")

        for col in selected_columns:
            if col not in df_named.columns:
                df_named[col] = np.nan

        df_selected = df_named[selected_columns].copy()

        # Convert categorical columns to numerical
        for col in UNIVERSAL_FEATURES:
            if df_selected[col].dtype == "object":
                print(f"Converting categorical column {col} to numerical in {source_name}")
                
                # First, try to extract year numbers from strings like "year 1", "Year 1", etc.
                if col == "Stress" and source_name == "D2-Cultural":
                    # Extract year number from strings like "year 1", "Year 1", "year 2", etc.
                    # Use regex to extract the number after "year"
                    df_selected[col] = df_selected[col].apply(
                        lambda x: re.search(r'year\s*(\d+)', str(x).lower())
                    ).apply(lambda m: int(m.group(1)) if m else np.nan)
                else:
                    # Standard categorical conversion
                    df_selected[col] = df_selected[col].astype(str).str.lower().replace(
                        {
                            "yes": 1,
                            "no": 0,
                            "often": 1,
                            "rarely": 0,
                            "sometimes": 0.5,
                            "maybe": 0.5,
                            "most of the time": 1,
                            "never": 0,
                            "always": 1,
                            "not sure": 0.5,
                            "high": 1,
                            "low": 0,
                            "medium": 0.5,
                            "somewhat easy": 0.5,
                            "somewhat difficult": 0.5,
                            "very difficult": 1,
                            "very easy": 0,
                            # Year mappings (for D2-Cultural Stress column)
                            "year 1": 1,
                            "year 2": 2,
                            "year 3": 3,
                            "year 4": 4,
                        }
                    )
                    df_selected[col] = pd.to_numeric(df_selected[col], errors="coerce")

        # OUTLIER DETECTION AND REMOVAL (IQR-based, per-dataset, per-feature)
        print(f"\nOutlier Detection for {source_name}...")

        for feature in UNIVERSAL_FEATURES:
            # Work on a numeric copy of the column
            values = pd.to_numeric(df_selected[feature], errors="coerce")

            # Skip if all NaN or constant
            if values.nunique(dropna=True) < 2:
                continue

            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1

            if IQR == 0:
                continue

            # IQR bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Domain-specific bounds for D4-Tech Age→Stress
            if source_name == "D4-Tech" and feature == "Stress":
                # Reasonable working range for tech workers
                lower_bound = max(lower_bound, 18)
                upper_bound = min(upper_bound, 80)

            outliers = (values < lower_bound) | (values > upper_bound)
            n_outliers = outliers.sum()

            if n_outliers > 0:
                print(f"  {feature}: Found {n_outliers} outlier(s)")
                print(f"    Bounds: {lower_bound:.2f} to {upper_bound:.2f}")
                unique_outliers = values[outliers].unique()
                for val in sorted(unique_outliers):
                    count_val = (values == val).sum()
                    print(f"    Value {val}: {count_val} row(s)")
                print(f"    → Removing {n_outliers} outlier row(s)")

                # Apply mask back to df_selected
                df_selected = df_selected[~outliers].reset_index(drop=True)

        # Handle missing values (fill with mean of THIS dataset)
        print(f"\nData Cleaning: Handling missing values for {source_name}")
        for feature in UNIVERSAL_FEATURES:
            mean_value = df_selected[feature].mean(skipna=True)
            missing_count = df_selected[feature].isna().sum()
            if missing_count > 0:
                print(f"  {feature}: Filling {missing_count} missing values with mean={mean_value:.4f}")
            df_selected[feature] = df_selected[feature].fillna(mean_value)

        # SPLIT INTO TRAIN AND TEST SETS (80/20 split, random seed=42)
        print(f"\nSplitting {source_name} into train and test sets (80/20 split, seed=42)")
        train_data, test_data = train_test_split(df_selected, test_size=0.2, random_state=42)

        # NORMALIZE TRAIN AND TEST SETS SEPARATELY
        print(f"\nNormalizing {source_name} features using Z-score normalization (train/test split)")
        scaler = StandardScaler()
        train_data[UNIVERSAL_FEATURES] = scaler.fit_transform(train_data[UNIVERSAL_FEATURES])
        test_data[UNIVERSAL_FEATURES] = scaler.transform(test_data[UNIVERSAL_FEATURES])

        # Save scaler for this dataset
        scaler_path = f"{source_name.replace('-', '_')}_scaler.joblib"
        joblib.dump(scaler, scaler_path)
        print(f"  Scaler saved as '{scaler_path}'")

        # Print normalization stats for train data
        print(f"  Normalized feature statistics (train data):")
        for feat in UNIVERSAL_FEATURES:
            mean_val = train_data[feat].mean()
            std_val = train_data[feat].std()
            print(f"    {feat}: mean={mean_val:.6f}, std={std_val:.6f}")

        # Save processed train and test datasets
        train_output_file = OUTPUT_PATHS[source_name].replace(".csv", "_train.csv")
        test_output_file = OUTPUT_PATHS[source_name].replace(".csv", "_test.csv")
        train_data.to_csv(train_output_file, index=False)
        test_data.to_csv(test_output_file, index=False)
        print(f"   {source_name} train dataset saved as '{train_output_file}'")
        print(f"   {source_name} test dataset saved as '{test_output_file}'")

        # Update processed_datasets dictionary
        processed_datasets[source_name] = {
            "train": train_output_file,
            "test": test_output_file,
        }

    except Exception as exc:
        print(f"Error loading {source_name} dataset: {exc}")
        import traceback
        traceback.print_exc()
        missing_sources.append(source_name)

# Final check for processed datasets
if not processed_datasets:
    print("\nNo datasets were processed. Exiting.")
    print("\nProcessed datasets:", processed_datasets)  # Debugging line to verify contents
    sys.exit(1)

# Alternative check for processed files
processed_files = [OUTPUT_PATHS[name].replace(".csv", "_train.csv") for name in FILE_PATHS.keys()]
if not any(Path(file).exists() for file in processed_files):
    print("\nNo processed files found. Exiting.")
    sys.exit(1)

if missing_sources:
    print("\n⚠ Datasets with missing files or errors:")
    for name in missing_sources:
        print(f"  - {name}")

# Combine all processed train datasets (already normalized separately)
print(f"\n{'='*60}")
print("Combining all processed train datasets")
print(f"{'='*60}")
train_datasets = [pd.read_csv(OUTPUT_PATHS[name].replace(".csv", "_train.csv")) for name in FILE_PATHS.keys() if name not in missing_sources]
df_combined_train = pd.concat(train_datasets, ignore_index=True)
print(f"Combined train dataset shape: {df_combined_train.shape}")

# Save fused train dataset
fused_train_output = "fused_mental_health_train_dataset.csv"
df_combined_train.to_csv(fused_train_output, index=False)
print(f"\n✓ Fused train dataset saved as '{fused_train_output}'")

print(f"\n{'='*60}")
print("Data processing complete!")
print(f"{'='*60}")
print(f"\nProcessed datasets:")
for source_name in FILE_PATHS.keys():
    print(f"  ✓ {source_name}: {OUTPUT_PATHS[source_name]} (train/test split)")
print(f"\nNote: Each dataset was normalized separately to preserve feature distributions.")