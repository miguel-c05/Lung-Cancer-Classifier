
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import argparse

def low_variance_filter(df, threshold_type='median_percentage', threshold_value=0.01):
    print("Starting low-variance filtering...")
    variances = df.var()

    if threshold_type == 'median_percentage':
        median_variance = variances.median()
        variance_threshold = median_variance * threshold_value
    elif threshold_type == 'absolute':
        variance_threshold = threshold_value
    else:
        raise ValueError("threshold_type must be 'median_percentage' or 'absolute'")

    low_variance_features = variances[variances < variance_threshold].index.tolist()
    df_filtered = df.drop(columns=low_variance_features)
    print(f"Original number of features: {df.shape[1]}")
    print(f"Number of low-variance features dropped: {len(low_variance_features)}")
    print(f"Remaining number of features: {df_filtered.shape[1]}")
    return df_filtered, low_variance_features, variance_threshold

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


def normalization(df_input):
    print("Starting normalization...")
    # Assuming the first two columns are 'patient_id' and 'nodule_id' and should not be scaled
    if 'patient_id' in df_input.columns and 'nodule_id' in df_input.columns:
        id_cols = df_input[['patient_id', 'nodule_id']].copy()
        df_features = df_input.drop(columns=['patient_id', 'nodule_id'])
    else:
        id_cols = pd.DataFrame(index=df_input.index) # Empty DataFrame if no ID columns
        df_features = df_input.copy()

    # Drop columns with NaN values before scaling
    df_features_cleaned = df_features.dropna(axis=1)

    scaler_cnn = StandardScaler()
    cnn_scaled_features = pd.DataFrame(scaler_cnn.fit_transform(df_features_cleaned), columns=df_features_cleaned.columns, index=df_features_cleaned.index)

    # Concatenate ID columns back if they existed
    if not id_cols.empty:
        cnn_scaled = pd.concat([id_cols, cnn_scaled_features], axis=1)
    else:
        cnn_scaled = cnn_scaled_features

    print("Normalization complete.")
    return cnn_scaled

def dimensionality_analysis(df_input, pca_variance_threshold=0.999):
    print("\nApplying PCA...")

    if 'patient_id' in df_input.columns and 'nodule_id' in df_input.columns:
        df_ids = df_input[['patient_id', 'nodule_id']].copy()
        df_features = df_input.drop(columns=['patient_id', 'nodule_id'])
    else:
        df_ids = pd.DataFrame(index=df_input.index)
        df_features = df_input.copy()

    # Drop columns with NaN values if any remain
    df_features_cleaned = df_features.dropna(axis=1)

    pca = PCA(n_components=pca_variance_threshold)
    pca_result_array = pca.fit_transform(df_features_cleaned)

    # Create a DataFrame for PCA results with meaningful column names
    pca_columns = [f'PC_{i+1}' for i in range(pca_result_array.shape[1])]
    pca_result_df = pd.DataFrame(pca_result_array, columns=pca_columns, index=df_features_cleaned.index)

    if not df_ids.empty:
        pca_result_df = pd.concat([df_ids, pca_result_df], axis=1)

    print(f"Original number of features: {df_features_cleaned.shape[1]}")
    print(f"Number of PCA components retained: {pca_result_array.shape[1]}")
    print(f"Total explained variance by PCA: {pca.explained_variance_ratio_.sum():.4f}")

    # Plot cumulative explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Cumulative Explained Variance')
    plt.grid(True)
    if not os.path.exists('pca_eda_plots'):
        os.makedirs('pca_eda_plots')
    plt.savefig('pca_eda_plots/pca_explained_variance.png')
    plt.close()

    print("PCA complete.")
    return pca_result_df

def integrated_eda_pipeline(input_csv_path, output_scaled_csv_name="densenet_scaled_features.csv", output_pca_csv_name="pca_features.csv"):
    print(f"Starting integrated EDA pipeline for {input_csv_path}...")

    # Load the original dataset
    df_original = pd.read_csv(input_csv_path)
    print(f"Original data shape: {df_original.shape}")

    # Separate ID columns if they exist
    id_columns = []
    if 'patient_id' in df_original.columns:
        id_columns.append('patient_id')
    if 'nodule_id' in df_original.columns:
        id_columns.append('nodule_id')

    if id_columns:
        df_ids = df_original[id_columns].copy()
        df_features = df_original.drop(columns=id_columns)
    else:
        df_ids = pd.DataFrame(index=df_original.index) # Empty DataFrame
        df_features = df_original.copy()

    # Drop columns with NaN values
    df_features_cleaned = df_features.dropna(axis=1)
    
    # --- Step 1: Low-Variance Filtering ---
    df_filtered_lv, _, _ = low_variance_filter(df_features_cleaned.copy(), threshold_type='median_percentage', threshold_value=0.01)

    # --- Step 2: Correlation Clustering ---
    df_final_filtered_corr, _ = correlation_filter_clustering(df_filtered_lv.copy(), correlation_threshold=0.9)

    # --- Step 3: Normalization (Scaling) ---
    # Re-attach IDs for normalization function if needed, then re-separate for scaling features only
    if id_columns:
        df_to_normalize = pd.concat([df_ids, df_final_filtered_corr], axis=1)
    else:
        df_to_normalize = df_final_filtered_corr

    df_scaled = normalization(df_to_normalize)

    # Save the scaled CSV
    df_scaled.to_csv(output_scaled_csv_name, index=False)
    print(f"Scaled features saved to {output_scaled_csv_name}")

    # --- Step 4: PCA ---
    df_pca_result = dimensionality_analysis(df_scaled.copy())

    # Save the PCA result
    df_pca_result.to_csv(output_pca_csv_name, index=False)
    print(f"PCA reduced features saved to {output_pca_csv_name}")

    print("Integrated EDA pipeline finished.")
    return df_pca_result

if __name__ == "__main__":
    input_dir = r"D:\aulas\lAB_iacd\lung-cancer-classifier\Densenet_CSVs"
    output_dir = r"D:\aulas\lAB_iacd\lung-cancer-classifier\EDA1\densenet_scaled"
    output_dir_pca = r"D:\aulas\lAB_iacd\lung-cancer-classifier\EDA1\EDA1\pca_features"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_pca, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(".csv"):
            continue
        # Skip already-processed files or outputs
        if fname.lower().endswith("_scaled.csv") or fname.lower().endswith("_pca.csv"):
            continue

        input_file = os.path.join(input_dir, fname)
        base = os.path.splitext(fname)[0]
        scaled_path = os.path.join(output_dir, base + "_scaled.csv")
        pca_path = os.path.join(output_dir_pca, base + "_pca.csv")

        """
        # If both outputs already exist, skip
        if os.path.exists(scaled_path) and os.path.exists(pca_path):
            print(f"Skipping {fname}: outputs already exist.")
            continue
        """

        print(f"\nProcessing file: {input_file} -> {scaled_path}, {pca_path}")
        try:
            integrated_eda_pipeline(input_file, output_scaled_csv_name=scaled_path, output_pca_csv_name=pca_path)
        except Exception as e:
            print(f"Error processing {input_file}: {e}")


