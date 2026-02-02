import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.cluster import KMeans

# Sampling1: Simple Random Sampling
def simple_random_sampling(df, frac=0.8):
    return df.sample(frac=frac, random_state=42)

# Sampling2: Systematic Sampling
def systematic_sampling(df, step=2):
    return df.iloc[::step, :]

# Sampling3: Stratified Sampling
def stratified_sampling(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return pd.concat([X_train, y_train], axis=1)

# Sampling4: Cluster Sampling (FIXED)
def cluster_sampling(df, n_clusters=5, clusters_to_select=3):
    X = df.drop("Class", axis=1)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    df = df.copy()
    df["Cluster"] = kmeans.fit_predict(X)

    # Select MULTIPLE clusters
    selected_clusters = np.random.choice(
        df["Cluster"].unique(),
        size=clusters_to_select,
        replace=False
    )

    sampled_df = df[df["Cluster"].isin(selected_clusters)]
    return sampled_df.drop("Cluster", axis=1)

# Sampling5: Bootstrap Sampling
def bootstrap_sampling(df):
    return resample(df, replace=True, n_samples=len(df), random_state=42)
