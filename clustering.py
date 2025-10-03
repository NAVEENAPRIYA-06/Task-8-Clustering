import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
FILE_NAME = 'Mall_Customers.csv' 
MAX_K = 11 
OPTIMAL_K = 5 
def load_data(file_name):
    """Loads the dataset and performs initial check."""
    print(f"Loading data from {file_name}...")
    try:
        df = pd.read_csv(file_name)
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found. Check the path and file name.")
        return None

def preprocess_data(df):
    """Performs feature selection and data scaling."""
    X_unscaled = df[['Annual Income (k$)', 'Spending Score (1-100)']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_unscaled)
    
    print("\nFeatures Selected & Data Scaled successfully.")
    
    return X_unscaled, X_scaled

def plot_elbow_method(X_scaled):
    """Calculates WCSS for different K values and plots the Elbow curve."""
    
    wcss = []
    # Test K from 1 up to MAX_K-1
    for i in range(1, MAX_K):
        kmeans = KMeans(n_clusters=i, 
                        init='k-means++', # Smart initialization
                        max_iter=300, 
                        n_init=10, 
                        random_state=42)
        kmeans.fit(X_scaled)
        # inertia_ is the WCSS (Within-Cluster Sum of Squares)
        wcss.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, MAX_K), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('WCSS (Inertia)')
    plt.grid(True)
    plt.xticks(range(1, MAX_K))
    plt.show()
    
    print(f"\nElbow Method plot generated. Look for the 'elbow' point.")
def fit_and_evaluate_kmeans(X_scaled, X_unscaled, k):
    """Fits K-Means with the optimal K, evaluates, and assigns labels."""
    
    print(f"\n--- 3. Fitting K-Means with Optimal K = {k} ---")
    
    kmeans = KMeans(n_clusters=k, 
                    init='k-means++', 
                    max_iter=300, 
                    n_init=10, 
                    random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    X_unscaled['Cluster'] = cluster_labels
    
    # Get cluster centers (in the scaled data space)
    centers_scaled = kmeans.cluster_centers_
    score = silhouette_score(X_scaled, cluster_labels)
    
    print(f"Silhouette Score for K={k}: {score:.4f}")
    
    return X_unscaled, centers_scaled
def visualize_clusters(X_unscaled, k):
    """Visualizes the clusters using the original, unscaled data."""
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Annual Income (k$)', 
                    y='Spending Score (1-100)', 
                    hue='Cluster', 
                    data=X_unscaled, 
                    palette='viridis', 
                    s=100, 
                    alpha=0.8, 
                    legend='full')
    
    plt.title(f'Customer Segments (K={k})')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    if k == 5:
        plt.text(75, 20, 'Target: High Income, Low Spend', fontsize=9, color='red', ha='left')
        plt.text(25, 80, 'Target: Low Income, High Spend', fontsize=9, color='red', ha='right')
        plt.text(80, 80, 'Target: High Income, High Spend', fontsize=12, color='green', ha='center', weight='bold')
        plt.text(25, 20, 'Target: Low Income, Low Spend', fontsize=9, color='red', ha='right')
        plt.text(50, 50, 'Baseline: Mid Income, Mid Spend', fontsize=9, color='gray', ha='center')

    plt.grid(True)
    plt.show()
    
    print("\nCluster visualization complete. The 5 distinct groups are visible.")

if __name__ == "__main__":
    
    # 1. Load and Preprocess Data
    mall_df = load_data(FILE_NAME)
    
    if mall_df is not None:
        X_unscaled, X_scaled = preprocess_data(mall_df)
        
        # 2. Elbow Method (for determining K)
        print("\n--- 2. Starting Elbow Method Analysis ---")
        plot_elbow_method(X_scaled)
        
        # 3. Fit K-Means, Evaluate, and Assign Labels
        # We manually set K=5 here based on the expected Elbow plot result.
        X_with_labels, centers = fit_and_evaluate_kmeans(X_scaled, X_unscaled.copy(), OPTIMAL_K)
        
        # 4. Visualization
        visualize_clusters(X_with_labels, OPTIMAL_K)
        
        # Display the characteristics of each cluster (optional but very useful)
        print("\n--- Cluster Summary (Mean values) ---")
        cluster_summary = X_with_labels.groupby('Cluster').mean()
        print(cluster_summary)

        print("\nK-Means Clustering Task Completed Successfully! ")