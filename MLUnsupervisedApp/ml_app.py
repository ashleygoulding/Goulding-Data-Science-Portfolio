# === Import required libraries ===
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

# === Load sample dataset (Iris or Wine) ===
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

# === Standardize the data so all features are on the same scale ===
def standardize_data(df):
    scaler = StandardScaler()
    return scaler.fit_transform(df)

# === Elbow plot to help determine optimal number of clusters in K-Means ===
def plot_elbow(X):
    distortions = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        distortions.append(kmeans.inertia_)
    plt.figure()
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k (Number of Clusters)')
    plt.ylabel('Distortion (Inertia)')
    plt.title('Elbow Method For Optimal k')
    st.pyplot(plt)

# === Visualize clusters after PCA dimensionality reduction ===
def plot_pca_clusters(X, labels, title):
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    plt.figure()
    sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=labels, palette="Set2")
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    st.pyplot(plt)

# === Draw dendrogram for hierarchical clustering ===
def plot_dendrogram(X, method):
    Z = linkage(X, method=method)
    plt.figure(figsize=(10, 5))
    dendrogram(Z, truncate_mode="level", p=5)
    plt.title("Hierarchical Clustering Dendrogram")
    st.pyplot(plt)

# === Show how much variance each PCA component explains ===
def plot_pca_variance(pca):
    explained_var = pca.explained_variance_ratio_
    plt.figure()
    plt.plot(range(1, len(explained_var)+1), explained_var, marker='o')
    plt.title("Explained Variance Ratio by Component")
    plt.xlabel("Principal Component")
    plt.ylabel("Proportion of Variance Explained")
    st.pyplot(plt)

# ================================
# STREAMLIT INTERFACE STARTS HERE
# ================================

# === Main title and description ===
st.title("🧠 Unsupervised Machine Learning Explorer")

st.markdown("""
Welcome to the Unsupervised ML Explorer!

This tool allows you to explore three key unsupervised learning techniques:

- **K-Means Clustering**
- **Hierarchical Clustering**
- **Principal Component Analysis (PCA)**

You can upload your own dataset or use one of the built-in ones. Visualizations and metrics will guide you through interpreting the results.

---
""")

# === Sidebar for user controls ===
st.sidebar.header("📁 Dataset Selection")

# === Dataset Upload or Selection ===
uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type="csv")
sample_dataset = st.sidebar.selectbox("Or select a sample dataset", ["None", "Iris", "Wine"])

# === Load selected dataset ===
if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif sample_dataset != "None":
    df = load_sample_dataset(sample_dataset)
else:
    st.warning("📂 Please upload a dataset or select a sample one from the sidebar.")
    st.stop()

# === Show dataset preview ===
st.subheader("📊 Dataset Preview")
st.markdown("Here’s the first few rows of your dataset. Make sure it’s clean and contains mostly numeric data for clustering.")
st.write(df.head())

# === Feature Selection Section ===
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
selected_features = st.sidebar.multiselect("Select features for analysis", options=numeric_columns, default=numeric_columns)

if not selected_features:
    st.info("ℹ️ No features selected. Using all numeric columns by default.")
    selected_features = numeric_columns

# === Standardize the data ===
X = standardize_data(df[selected_features])

# === Method Selection ===
st.sidebar.header("⚙️ Select Unsupervised Method")
method = st.sidebar.selectbox("Choose a technique to apply", [
    "K-Means Clustering", 
    "Hierarchical Clustering", 
    "Principal Component Analysis"
])

# ================================
# K-MEANS CLUSTERING
# ================================
if method == "K-Means Clustering":
    st.subheader("📈 K-Means Clustering")

    st.markdown("""
K-Means attempts to partition the dataset into **k distinct, non-overlapping clusters** based on feature similarity.

Use the slider to adjust `k` — the number of clusters. Look at the **Silhouette Score** and **Elbow Plot** to find the best `k`.

- A **higher silhouette score (close to 1)** means well-separated clusters.
- The **elbow point** in the Elbow Plot suggests the optimal k.

    """)

    # Cluster number selector
    k = st.sidebar.slider("Number of clusters (k)", 2, 10, 3)

    # Train model and get cluster labels
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(X)
    silhouette = silhouette_score(X, labels)

    # Results
    st.write(f"**Silhouette Score:** {silhouette:.3f} — closer to 1 means better-defined clusters.")

    st.markdown("#### 🔍 Cluster Visualization (via PCA)")
    st.markdown("We use PCA to reduce to 2D and show cluster separations visually.")
    plot_pca_clusters(X, labels, "PCA-based Cluster Visualization")

    st.markdown("#### 📉 Elbow Method Plot")
    st.markdown("""
    This plot helps you choose the **optimal number of clusters (k)** in K-Means.

    - The **x-axis** shows different values of `k` (clusters).
    - The **y-axis** shows the **inertia**, or within-cluster sum of squares (how compact clusters are).

    **How to interpret:**
    - You're looking for the **'elbow point'** — where the inertia decreases sharply and then flattens out.
    - This point balances compactness and complexity: adding more clusters beyond this gives diminishing returns.

    **Why it matters:**  
    Too few clusters = underfitting (merging distinct groups),  
    Too many clusters = overfitting (splitting true groups unnecessarily).
    """)

# ================================
# HIERARCHICAL CLUSTERING
# ================================
elif method == "Hierarchical Clustering":
    st.subheader("🔗 Hierarchical Clustering")

    st.markdown("""
Hierarchical Clustering builds a tree of clusters without needing a fixed `k`. 
We still cut the tree into a specific number of clusters for evaluation.

- Choose a **linkage method** (how distances are calculated between clusters).
- Adjust `k` (number of clusters) and examine the **dendrogram**.
- Higher **silhouette score** still means better-defined clusters.

**Linkage options:**
- `ward`: minimizes variance within clusters
- `complete`: max distance between clusters
- `average`: average distance
- `single`: closest point distance
""")

    linkage_method = st.sidebar.selectbox("Linkage Method", ["ward", "complete", "average", "single"])
    k = st.sidebar.slider("Number of clusters", 2, 10, 3)

    model = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
    labels = model.fit_predict(X)
    silhouette = silhouette_score(X, labels)

    st.write(f"**Silhouette Score:** {silhouette:.3f}")

    st.markdown("#### 🔍 Cluster Visualization (via PCA)")
    st.markdown("""
    This plot reduces your data to 2D using **Principal Component Analysis (PCA)**, then colors points based on the clustering result.

    **Why use PCA here?**
    - High-dimensional data can't be visualized easily.
    - PCA projects it onto 2D while preserving variance.
    - It lets us *see* how well-separated the clusters are.

    **How to interpret:**
    - Each point is a data row.
    - Clusters should appear as distinct, tight groups.
    - Overlapping clusters may indicate poor separation or that more informative features are needed.
    """)
    plot_pca_clusters(X, labels, "PCA-based Cluster Visualization")

    st.markdown("#### 🌲 Dendrogram")
    st.markdown("""
    The dendrogram shows the **hierarchical merging process** between observations in the dataset.

    **How to interpret:**
    - **Each leaf** is a data point.
    - **Merges** (horizontal lines) show which points/clusters are grouped together and at what **dissimilarity level**.
    - **Vertical height** of lines indicates how far apart clusters are.
    - Cutting the dendrogram at a given height gives you the desired number of clusters.

    **Look for:**
    - **Large vertical gaps** between horizontal lines — these suggest natural groupings.
    - Flat merges near the bottom mean similar data points; tall merges mean bigger jumps in dissimilarity.

    This helps you understand both structure and how many clusters make sense in your data.
    """)
    plot_dendrogram(X, linkage_method)

# ================================
# PRINCIPAL COMPONENT ANALYSIS
# ================================
elif method == "Principal Component Analysis":
    st.subheader("📉 Principal Component Analysis (PCA)")

    st.markdown("""
PCA reduces dimensionality by projecting data into a smaller number of components while retaining the most variance.

This is helpful for:
- **Visualization**
- **Noise reduction**
- **Feature compression**

Use the slider to choose how many components to retain.
""")

    # Component selector
    n_components = st.sidebar.slider("Number of components", 2, min(10, X.shape[1]), 2)

    # Perform PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    explained_var = pca.explained_variance_ratio_

    st.markdown("#### 🔢 Explained Variance by Component")
    st.markdown("""
This shows how much variance (information) is captured by each component. Try to retain enough components to explain ~90% of the variance.
""")
    st.write(np.round(explained_var, 3))

    st.markdown("#### 📊 2D PCA Projection")
    st.markdown("This scatter plot shows the first two principal components — a compressed view of your data.")
    plt.figure()
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1])
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("2D PCA Projection")
    st.pyplot(plt)

    st.markdown("#### 📈 Variance Explained Plot")
    st.markdown("""
    This graph shows how much of the total variance (information) in your dataset is captured by each principal component.

    **How to interpret:**
    - The **x-axis** is the principal component index.
    - The **y-axis** is the proportion of total variance explained.

    **Look for:**
    - The **elbow point** here helps decide how many components to keep.
    - Typically, choose enough components to explain **80–95%** of the variance.

    **Why it matters:**
    - Retaining too few components may lose important information.
    - Keeping too many defeats the purpose of dimensionality reduction.

    Use this plot alongside your knowledge of the data to decide how much complexity is needed.
    """)
    plot_pca_variance(pca)