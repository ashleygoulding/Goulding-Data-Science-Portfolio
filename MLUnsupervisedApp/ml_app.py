import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# App Title
st.title("🧠 Unsupervised Machine Learning Explorer")

# Sidebar options
st.sidebar.header("🔧 Model Controls")

# File uploader and sample dataset options
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type="csv")
sample_dataset = st.sidebar.selectbox("Or select a sample dataset", ["None", "Iris", "Wine"])

# Load dataset
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif sample_dataset == "Iris":
    from sklearn.datasets import load_iris
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
elif sample_dataset == "Wine":
    from sklearn.datasets import load_wine
    data = load_wine()
    df = pd.DataFrame(data.data, columns=data.feature_names)
else:
    st.warning("Please upload a dataset or select a sample one.")
    st.stop()

# Show data preview
st.subheader("📊 Dataset Preview")
st.write(df.head())

# Technique selection
method = st.sidebar.selectbox("Select Method", ["K-Means Clustering", "Hierarchical Clustering", "Principal Component Analysis"])

# Standardize data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(df)

# Helper for plotting
def plot_elbow(X):
    distortions = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        distortions.append(kmeans.inertia_)
    plt.figure()
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion (Inertia)')
    plt.title('Elbow Method For Optimal k')
    st.pyplot(plt)

# K-Means Clustering
if method == "K-Means Clustering":
    k = st.sidebar.slider("Number of clusters (k)", 2, 10, 3)
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(X)
    silhouette = silhouette_score(X, labels)
    st.subheader("📈 K-Means Clustering Results")
    st.write(f"Silhouette Score: {silhouette:.3f}")

    # PCA for plotting
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    plt.figure()
    sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=labels, palette="Set2")
    plt.title("Cluster Visualization (PCA Reduced)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    st.pyplot(plt)

    st.subheader("📉 Elbow Plot")
    plot_elbow(X)

# Hierarchical Clustering
elif method == "Hierarchical Clustering":
    linkage_method = st.sidebar.selectbox("Linkage Method", ["ward", "complete", "average", "single"])
    k = st.sidebar.slider("Number of clusters", 2, 10, 3)
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
    labels = model.fit_predict(X)
    silhouette = silhouette_score(X, labels)
    st.subheader("🔗 Hierarchical Clustering Results")
    st.write(f"Silhouette Score: {silhouette:.3f}")

    # PCA for plotting
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    plt.figure()
    sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=labels, palette="Set1")
    plt.title("Cluster Visualization (PCA Reduced)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    st.pyplot(plt)

    # Dendrogram
    st.subheader("🌲 Dendrogram")
    Z = linkage(X, method=linkage_method)
    plt.figure(figsize=(10, 5))
    dendrogram(Z, truncate_mode="level", p=5)
    plt.title("Hierarchical Clustering Dendrogram")
    st.pyplot(plt)

# PCA
elif method == "Principal Component Analysis":
    n_components = st.sidebar.slider("Number of components", 2, min(10, X.shape[1]), 2)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    explained_var = pca.explained_variance_ratio_
    st.subheader("📉 PCA Results")
    st.write("Explained Variance by Component:", explained_var.round(3))

    # Plotting
    plt.figure()
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Scatter Plot")
    st.pyplot(plt)

    # Variance explained plot
    plt.figure()
    plt.plot(range(1, n_components+1), explained_var, marker='o')
    plt.title("Explained Variance Ratio")
    plt.xlabel("Principal Component")
    plt.ylabel("Proportion of Variance Explained")
    st.pyplot(plt)

st.markdown("---")
st.markdown("Developed with ❤️ for intuitive unsupervised learning exploration.")
