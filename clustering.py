
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

# --- Configuration ---
FILE_NAME = 'Mall_Customers.csv'
def load_data(file_name):
    """Loads the dataset."""
    print(f"Loading data from {file_name}...")
    try:
        df = pd.read_csv(file_name)
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found. Check the path and file name.")
        return None

def preprocess_data(df):
    """Performs necessary cleaning and feature selection."""
    
    print("\n--- Initial Data Info ---")
    df.info()
    
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    
    print("\nFeatures selected for clustering (X):")
    print(X.head())

    # 2. Scaling the data: K-Means is sensitive to scale, so we use StandardScaler.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("\nData scaled successfully.")
    
    return X, X_scaled

# --- Main Execution ---
if __name__ == "__main__":
    
    # 1. Load Data
    mall_df = load_data(FILE_NAME)
    
    if mall_df is not None:
        
        # 2. Preprocess Data (Feature selection and scaling)
        X_unscaled, X_scaled = preprocess_data(mall_df)
        
        # We will save the rest of the clustering logic for subsequent steps/commits!
        print("\nPreprocessing complete. Ready for Elbow Method.")