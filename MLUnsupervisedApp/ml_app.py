# === Import necessary libraries ===
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# scikit-learn: machine learning algorithms & utilities
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# scipy: used for creating hierarchical clustering dendrograms
from scipy.cluster.hierarchy import dendrogram, linkage


# ================================
# Helper Functions
# ================================

# Loads sample datasets from sklearn if user chooses one
def load_sample_dataset(name):
    if name == "Iris":
        from sklearn.datasets import load_iris
        data = load_iris()
    elif name == "Wine":
        from sklearn.datasets import load_wine
        data = load_wine()
    else:
        return None
    return pd.DataFrame(data.data, columns=data.feature_names)

# Standardizes features by removing the mean and scaling to unit variance
# This ensures all features contribute equally to distance-based models
def standardize_data(df):
    scaler = StandardScaler()
    return scaler.fit_transform(df)

# Elbow plot helps user visually identify the optimal number of clusters (K)
def plot_elbow(X):
    distortions = []  # List to hold inertia values for each k
    K = range(1, 11)  # Try values of k from 1 to 10
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        distortions.append(kmeans.inertia_)  # Inertia: within-cluster sum of squares
    plt.figure()
    plt.plot(K, distortions, 'bx-')  # Plot K vs inertia
    plt.xlabel('k (Number of Clusters)')
    plt.ylabel('Distortion (Inertia)')
    plt.title('Elbow Method For Optimal k')
    st.pyplot(plt)

# Reduces data to 2D using PCA and plots the resulting cluster assignments
def plot_pca_clusters(X, labels, title):
    pca = PCA(n_components=2)  # Reduce to 2 principal components for visualization
    components = pca.fit_transform(X)
    plt.figure()
    sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=labels, palette="Set2")
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    st.pyplot(plt)

# Creates a dendrogram for visualizing hierarchical clustering structure
def plot_dendrogram(X, method):
    Z = linkage(X, method=method)  # Computes hierarchical clustering
    plt.figure(figsize=(10, 5))
    dendrogram(Z, truncate_mode="level", p=5)  # Show last p merges
    plt.title("Hierarchical Clustering Dendrogram")
    st.pyplot(plt)

# Plots how much variance is explained by each principal component
def plot_pca_variance(pca):
    explained_var = pca.explained_variance_ratio_
    plt.figure()
    plt.plot(range(1, len(explained_var)+1), explained_var, marker='o')
    plt.title("Explained Variance Ratio")
    plt.xlabel("Principal Component")
    plt.ylabel("Proportion of Variance Explained")
    st.pyplot(plt)


# ================================
# Streamlit App Interface
# ================================

# Title of the application
st.title("🧠 Unsupervised Machine Learning Explorer")

# Sidebar header for user controls
st.sidebar.header("🔧 Model Controls")

# === Dataset Uploading Section ===
# Users can either upload a CSV file or choose a sample dataset
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type="csv")
sample_dataset = st.sidebar.selectbox("Or select a sample dataset", ["None", "Iris", "Wine"])

# === Load Dataset ===
if uploaded_file:
    # If user uploads their own dataset
    df = pd.read_csv(uploaded_file)
elif sample_dataset != "None":
    # If user selects one of the built-in datasets
    df = load_sample_dataset(sample_dataset)
else:
    # If neither is provided, prompt user to upload or choose
    st.warning("Please upload a dataset or select a sample one.")
    st.stop()

# === Show Dataset Preview ===
st.subheader("📊 Dataset Preview")
st.write(df.head())  # Display first few rows of the dataset

# === Feature Selection ===
# Automatically selects only numeric columns for clustering/PCA
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

# Allow user to select which features (columns) to include
selected_features = st.sidebar.multiselect(
    "Feature Selection", 
    options=numeric_columns, 
    default=numeric_columns  # Select all numeric by default
)

# If no features are selected, default to all numeric columns
if not selected_features:
    st.info("No features selected. Using all numeric columns by default.")
    selected_features = numeric_columns

# === Preprocess the Data ===
# Standardize the selected numeric features
X = standardize_data(df[selected_features])

# === Algorithm Selection ===
# User chooses which unsupervised learning method to explore
method = st.sidebar.selectbox("Select Method", [
    "K-Means Clustering", 
    "Hierarchical Clustering", 
    "Principal Component Analysis"
])

# ================================
# K-MEANS CLUSTERING
# ================================
if method == "K-Means Clustering":
    # User selects number of clusters
    k = st.sidebar.slider("Number of clusters (k)", 2, 10, 3)
    
    # Apply K-Means clustering to standardized data
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(X)
    
    # Compute silhouette score to evaluate clustering performance
    silhouette = silhouette_score(X, labels)

    # === Display Results ===
    st.subheader("📈 K-Means Clustering Results")
    st.write(f"Silhouette Score: {silhouette:.3f} (Higher is better)")
    
    # Visualize clusters in 2D using PCA projection
    plot_pca_clusters(X, labels, "Cluster Visualization (PCA Reduced)")

    # Show elbow plot to guide user in selecting optimal k
    st.subheader("📉 Elbow Plot")
    plot_elbow(X)

# ================================
# HIERARCHICAL CLUSTERING
# ================================
elif method == "Hierarchical Clustering":
    # User selects linkage method and number of clusters
    linkage_method = st.sidebar.selectbox("Linkage Method", ["ward", "complete", "average", "single"])
    k = st.sidebar.slider("Number of clusters", 2, 10, 3)
    
    # Apply hierarchical clustering with chosen linkage method
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
    labels = model.fit_predict(X)
    
    # Compute silhouette score
    silhouette = silhouette_score(X, labels)

    # === Display Results ===
    st.subheader("🔗 Hierarchical Clustering Results")
    st.write(f"Silhouette Score: {silhouette:.3f}")
    
    # Visualize clustering results
    plot_pca_clusters(X, labels, "Cluster Visualization (PCA Reduced)")
    
    # Show the dendrogram to illustrate clustering hierarchy
    st.subheader("🌲 Dendrogram")
    plot_dendrogram(X, linkage_method)

# ================================
# PRINCIPAL COMPONENT ANALYSIS
# ================================
elif method == "Principal Component Analysis":
    # User selects number of components to retain
    n_components = st.sidebar.slider("Number of components", 2, min(10, X.shape[1]), 2)

    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    explained_var = pca.explained_variance_ratio_

    # === Display Results ===
    st.subheader("📉 PCA Results")
    st.write("Explained Variance by Component:", explained_var.round(3))

    # Show 2D PCA scatter plot
    plt.figure()
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Scatter Plot")
    st.pyplot(plt)

    # Show explained variance per component
    plot_pca_variance(pca)