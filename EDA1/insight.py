
import pandas as pd
#df = pd.read_csv("C:\\Users\\ruial\\OneDrive\\Ambiente de Trabalho\\Lab_IACD\\EDA\\dicom_densenet_features_final_filtered.csv")

#print(df.head())
#print(df.info())
#print(df.shape)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
"""
# Load your DataFrame
# Replace with your actual CSV file or DataFrame
#df = pd.read_csv("C:\\Users\\ruial\\lung-cancer-classifier\\Densenet_CSVs\\densenet_features.csv")

#df= pd.read_csv("C:\\Users\\ruial\\lung-cancer-classifier\\Densenet_CSVs\\densenet_features_nodule.csv")

#df = pd.read_csv("C:\\Users\\ruial\\lung-cancer-classifier\\densenet_bof_features.csv")

df = pd.read_csv(r"C:\\Users\\ruial\\lung-cancer-classifier\\Densenet_CSVs\\densenet_features_slices.csv")

# 1️ Check for NaN values in each column
nan_counts = df.isna().sum()
print("Number of NaNs per column:")
print(nan_counts[nan_counts > 0])

# 2️ Check for constant columns (only one unique value)
constant_cols = df.columns[df.nunique() <= 1].tolist()
print("\nColumns with constant values (to consider dropping):")
print(constant_cols)

# 3️ Optional: Check for columns that are all zeros
all_zero_cols = df.columns[(df == 0).all()].tolist()
print("\nColumns that are all zeros:")
print(all_zero_cols)


# 4️ Optional: Summary
print(f"\nTotal columns: {df.shape[1]}")
print(f"Columns with NaNs: {nan_counts[nan_counts > 0].shape[0]}")
print(f"Constant columns: {len(constant_cols)}")
print(f"All-zero columns: {len(all_zero_cols)}")"""

densenet_path = r"C:\Users\ruial\lung-cancer-classifier\Densenet_CSVs\densenet_features_nodulo.csv"
radiomics_path= r"C:\Users\ruial\lung-cancer-classifier\Radiomics_CSVs\radiomics_original.csv"

PLOTS_DIR = r'C:\Users\ruial\lung-cancer-classifier\EDA1\pre_eda_plots'


# 1. Load datasets
print(f"Loading DenseNet features from: {densenet_path}")
df_densenet = pd.read_csv(densenet_path)


print(f"Loading PyRadiomics features from: {radiomics_path}")
df_radiomics = pd.read_csv(radiomics_path)

print(f"\nInitial DenseNet DataFrame Shape: {df_densenet.shape}")
print(f"Initial Radiomics DataFrame Shape: {df_radiomics.shape}")
# Identify numeric radiomics features and compute statistics (ignoring NaNs)
os.makedirs(PLOTS_DIR, exist_ok=True)

num_cols = df_radiomics.select_dtypes(include=[np.number]).columns.tolist()
if not num_cols:
    print("No numeric columns found in radiomics DataFrame.")
else:
    # variance (population), min, max, domain, and non-null count
    variances = df_radiomics[num_cols].var(axis=0, skipna=True, ddof=0)
    mins = df_radiomics[num_cols].min(axis=0, skipna=True)
    maxs = df_radiomics[num_cols].max(axis=0, skipna=True)
    domains = maxs - mins
    non_nulls = df_radiomics[num_cols].count()

    summary = pd.DataFrame({
        "variance": variances,
        "min": mins,
        "max": maxs,
        "domain": domains,
        "non_null_count": non_nulls
    })

    # sort by variance for convenience
    summary_sorted = summary.sort_values("variance")

    # report lowest/highest variance features
    min_col = summary_sorted.index[0]
    max_col = summary_sorted.index[-1]
    print(f"Lowest variance feature: {min_col} -> {summary.loc[min_col,'variance']:.6g}, "
          f"min={summary.loc[min_col,'min']}, max={summary.loc[min_col,'max']}, "
          f"domain={summary.loc[min_col,'domain']}")
    print(f"Highest variance feature: {max_col} -> {summary.loc[max_col,'variance']:.6g}, "
          f"min={summary.loc[max_col,'min']}, max={summary.loc[max_col,'max']}, "
          f"domain={summary.loc[max_col,'domain']}")

    # save full table for inspection
    summary_path = os.path.join(PLOTS_DIR, "radiomics_features_variance_summary.csv")
    summary_sorted.to_csv(summary_path, index=True)
    print(f"Saved radiomics feature summary to: {summary_path}")

"""
# Plot for a few DenseNet features
plt.figure(figsize=(18, 5))
plt.suptitle("DenseNet Feature Distributions (Sample)", fontsize=16)
for i, col in enumerate(df_densenet.columns[:5]): # Plot first 5 features
    plt.subplot(1, 5, i + 1)
    sns.histplot(df_densenet[col].dropna(), kde=True, bins=30)
    plt.title(col)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
densenet_hist_path = os.path.join(PLOTS_DIR, 'densenet_feature_histograms.png')
plt.savefig(densenet_hist_path)
plt.close()
print(f"Saved DenseNet histograms to: {densenet_hist_path}")

# Plot for a few Radiomics features
plt.figure(figsize=(18, 5))
plt.suptitle("PyRadiomics Feature Distributions (Sample)", fontsize=16)
for i, col in enumerate(df_radiomics.columns[:5]): # Plot first 5 features
    plt.subplot(1, 5, i + 1)
    sns.histplot(df_radiomics[col].dropna(), kde=True, bins=30)
    plt.title(col)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
radiomics_hist_path = os.path.join(PLOTS_DIR, 'radiomics_feature_histograms.png')
plt.savefig(radiomics_hist_path)
plt.close()
print(f"Saved PyRadiomics histograms to: {radiomics_hist_path}")
"""