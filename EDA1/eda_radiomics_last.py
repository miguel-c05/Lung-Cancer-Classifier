import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import os


def correlation_filter_clustering(df, correlation_threshold=0.9, plot_prefix="combined"):
    """
    Performs correlation-based feature filtering using a custom clustering rule.
    A new feature is merged into an existing cluster only if it has correlation >= correlation_threshold
    with ALL features already in that cluster. This ensures tight, internally consistent clusters.

    Args:
        df (pd.DataFrame): The input DataFrame with features as columns.
        correlation_threshold (float): The absolute correlation threshold to define clusters.
                                     Features within a cluster will have a pairwise correlation above this threshold.
        plot_prefix (str): Prefix for saved plot filenames.

    Returns:
        pd.DataFrame: DataFrame with features selected by clustering-based correlation filtering.
        list: List of dropped column names.
    """
    print(f"\nStarting custom clustering-based correlation filtering for {plot_prefix} features...")
    df_copy = df.copy()

    # Calculate the absolute correlation matrix
    corr_matrix = df_copy.corr().abs()
    features = list(df_copy.columns)
    
    clusters = [] # Each element in clusters will be a list of feature names

    # Iterate through each feature to assign it to a cluster
    for feature in features:
        added = False
        # Try to add the feature to an existing cluster
        for cluster in clusters:
            # Check if the feature is highly correlated with ALL members of the current cluster
            if all(corr_matrix.loc[feature, member] >= correlation_threshold for member in cluster):
                cluster.append(feature)
                added = True
                break # Feature added, move to the next feature
        # If the feature couldn't be added to any existing cluster, start a new one
        if not added:
            clusters.append([feature])

    selected_features = []
    dropped_features = []

    # From each cluster, select a representative feature
    for cluster in clusters:
        if len(cluster) == 1:
            selected_features.append(cluster[0])
        else:
            # Select the feature with the highest mean absolute correlation to all other features in the original dataset
            #mean_abs_correlations = corr_matrix.loc[cluster, :].mean(axis=1)
            #representative_feature = mean_abs_correlations.idxmax()

            # Compute mean correlations only within the cluster
            cluster_corr = corr_matrix.loc[cluster, cluster]
            mean_corr_within_cluster = cluster_corr.mean(axis=1)
            representative_feature = mean_corr_within_cluster.idxmax()
            
            selected_features.append(representative_feature)
            dropped_features.extend([f for f in cluster if f != representative_feature])

    df_filtered = df_copy[selected_features]

    print(f"Original number of features: {df.shape[1]}")
    print(f"Number of features dropped by custom clustering-based correlation: {len(dropped_features)}")
    print(f"Remaining number of features: {df_filtered.shape[1]}")

    return df_filtered, dropped_features

def process_radiomics_csv(input_csv_path, output_csv_path, correlation_threshold=0.9):
    print(f"Starting integrated EDA for {input_csv_path}...")

    df_original = pd.read_csv(input_csv_path)
    
    # Drop columns that are all NaN, as per integrate_filtering_radiomics.py
    df_original = df_original.dropna(axis=1, how='all')

    # Identify patient_id and nodule_id columns dynamically
    id_columns = []
    if 'patient_id' in df_original.columns: id_columns.append('patient_id')
    if 'nodule_id' in df_original.columns: id_columns.append('nodule_id')
    if 'nodule_idx' in df_original.columns: id_columns.append('nodule_idx') # For normalization_merge.py compatibility

    temp_id_cols = df_original[id_columns].copy()
    df_features = df_original.drop(columns=id_columns)

    print(f"Original data shape: {df_features.shape}")

    # --- Step 1: Correlation Matrix Filtering ---
    print("\n--- Performing Correlation Matrix Filtering ---")
    df_filtered, _ = correlation_filter_clustering(
        df_features.copy(),
        correlation_threshold=correlation_threshold
    )
    print(f"Correlation filtering complete. Final remaining features: {df_filtered.shape[1]}")

    # --- Step 2: Normalization ---
    print("\n--- Performing Normalization (RobustScaler only) ---")

    scaler_radiomics = RobustScaler()
    radiomics_scaled = pd.DataFrame(scaler_radiomics.fit_transform(df_filtered), columns=df_filtered.columns)
    print(f"Normalization complete. Scaled data shape: {radiomics_scaled.shape}")

    # Re-attach ID columns
    df_final_scaled = pd.concat([temp_id_cols, radiomics_scaled], axis=1)

    # Save final scaled data
    df_final_scaled.to_csv(output_csv_path, index=False)
    print(f"Final scaled data saved to {output_csv_path}")
    print("Integrated EDA process finished.")



if __name__ == "__main__":
    # Example usage:
    # To make it executable for various CSV files, you can pass the input_csv_path as a command-line argument
    # For now, we'll use the provided radiomics_original.csv
    input_dir = "C:\\Users\\ruial\\lung-cancer-classifier\\Radiomics_CSVs"
    output_dir = "C:\\Users\\ruial\\lung-cancer-classifier\\EDA1\\radiomics_scaled_last"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all CSV files in the folder and process each
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".csv"):
            continue
        # Skip already-scaled outputs to avoid reprocessing
        if fname.lower().endswith("_scaled.csv"):
            continue

        input_file = os.path.join(input_dir, fname)
        output_file = os.path.join(output_dir, os.path.splitext(fname)[0] + "_scaled.csv")
        print(f"\nProcessing file: {input_file} -> {output_file}")
        try:
            process_radiomics_csv(input_file, output_file)
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
