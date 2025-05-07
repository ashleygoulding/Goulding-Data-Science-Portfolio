# Import required libraries 
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

# Load sample dataset (Iris or Wine) 
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

# Standardize the data so all features are on the same scale =
def standardize_data(df):
    scaler = StandardScaler()
    return scaler.fit_transform(df)

# Elbow plot to help determine optimal number of clusters in K-Means
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

# Visualize clusters after PCA dimensionality reduction
def plot_pca_clusters(X, labels, title):
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    plt.figure()
    sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=labels, palette="Set2")
    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    st.pyplot(plt)

# Draw dendrogram for hierarchical clustering
def plot_dendrogram(X, method):
    Z = linkage(X, method=method)
    plt.figure(figsize=(10, 5))
    dendrogram(Z, truncate_mode="level", p=5)
    plt.title("Hierarchical Clustering Dendrogram")
    st.pyplot(plt)

# Show how much variance each PCA component explains
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

# Main title and description
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

# Sidebar for user controls
st.sidebar.header("📁 Dataset Selection")

# Dataset Upload or Selection
uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type="csv")
sample_dataset = st.sidebar.selectbox("Or select a sample dataset", ["None", "Iris", "Wine"])

# Load selected dataset
if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif sample_dataset != "None":
    df = load_sample_dataset(sample_dataset)
else:
    st.warning("📂 Please upload a dataset or select a sample one from the sidebar.")
    st.stop()

# Handle missing data
if df.isnull().sum().any():
    st.warning("⚠️ Your dataset contains missing values. Please consider handling them.")
    action = st.radio("How would you like to handle missing data?", ["Remove rows", "Fill with mean", "Fill with median"], index=0)
    if action == "Remove rows":
        df = df.dropna()
    elif action == "Fill with mean":
        df = df.fillna(df.mean())
    elif action == "Fill with median":
        df = df.fillna(df.median())

# Show dataset preview
st.subheader("📊 Dataset Preview")
st.markdown("Here’s the first few rows of your dataset. Make sure it’s clean and contains mostly numeric data for clustering.")
st.write(df.head())

# Feature Selection Section
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
selected_features = st.sidebar.multiselect("Select features for analysis", options=numeric_columns, default=numeric_columns)

if not selected_features:
    st.info("ℹ️ No features selected. Using all numeric columns by default.")
    selected_features = numeric_columns

# Standardize the data
X = standardize_data(df[selected_features])

# Method Selection
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

    # Fit KMeans model
    kmeans = KMeans(n_clusters=k, random_state=0)
    clusters = kmeans.fit_predict(X)
    
    # Silhouette Score evaluation
    silhouette = silhouette_score(X, clusters)
    st.write(f"**Silhouette Score:** {silhouette:.3f}")

    # PCA for 2D visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # 1: KMeans clustering in PCA space
    st.markdown("### 🔵 K-Means Clustering (2D PCA Projection)")
    st.markdown(
        "This scatter plot shows how the K-Means algorithm grouped the observations using only the first two principal components. "
        "Each point represents a data sample, colored by its cluster assignment."
    )
    plt.figure(figsize=(8, 6))
    for i in range(k):
        plt.scatter(X_pca[clusters == i, 0], X_pca[clusters == i, 1],
                    alpha=0.7, edgecolor='k', s=60, label=f'Cluster {i}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('K-Means Clustering: 2D PCA Projection')
    plt.legend(loc='best')
    plt.grid(True)
    st.pyplot(plt)

    # 2: True Labels if available 
    if 'target' in df.columns:
        y = df['target'].values
        target_names = sorted(df['target'].unique())
        st.markdown("### 🟠 True Labels in PCA Space")
        st.markdown(
            "This scatter plot uses the actual target labels to show the ground truth distribution of data. "
            "By comparing this with the clustering output above, you can visually assess clustering quality."
        )
        plt.figure(figsize=(8, 6))
        colors = sns.color_palette("Set1", len(target_names))
        for i, label in enumerate(target_names):
            plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1],
                        alpha=0.7, edgecolor='k', s=60, label=f'Class {label}')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('True Labels: 2D PCA Projection')
        plt.legend(loc='best')
        plt.grid(True)
        st.pyplot(plt)

        # Evaluate match (note: label alignment is arbitrary)
        from sklearn.metrics import accuracy_score
        from scipy.stats import mode

        # Relabel clusters to best match ground truth using majority voting
        new_labels = np.zeros_like(clusters)
        for i in range(k):
            mask = clusters == i
            if np.sum(mask) == 0:
                continue
            new_labels[mask] = mode(y[mask], keepdims=True)[0]
        accuracy = accuracy_score(y, new_labels)
        st.write(f"**Adjusted Clustering Accuracy:** {accuracy * 100:.2f}%")
        st.markdown("Note: Cluster labels are permuted automatically to best match the true labels.")

    # 3: Elbow and Silhouette Evaluation ---
    st.markdown("### 📊 Selecting Optimal k")
    st.markdown("""
        The **Elbow Method** plot helps you choose the **optimal number of clusters (k)** in K-Means.

        - The **x-axis** shows different values of `k` (clusters).
        - The **y-axis** shows the **inertia**, or within-cluster sum of squares (how compact clusters are).

        **How to interpret:**
        - You're looking for the **'elbow point'** — where the inertia decreases sharply and then flattens out.
        - This point balances compactness and complexity: adding more clusters beyond this gives diminishing returns.

        **Why it matters:**  
        Too few clusters = underfitting (merging distinct groups),  
        Too many clusters = overfitting (splitting true groups unnecessarily).

        **Silhouette Score** measures how well-separated the resulting clusters are. Higher values indicate better-defined clusters.
        
        """)
    ks = range(2, 11)
    wcss = []
    silhouette_scores = []
    for i in ks:
        km = KMeans(n_clusters=i, random_state=42)
        km.fit(X)
        wcss.append(km.inertia_)
        labels = km.labels_
        silhouette_scores.append(silhouette_score(X, labels))

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Elbow plot
    ax[0].plot(ks, wcss, marker='o')
    ax[0].set_xlabel('Number of clusters (k)')
    ax[0].set_ylabel('Within-Cluster Sum of Squares (WCSS)')
    ax[0].set_title('Elbow Method for Optimal k')
    ax[0].grid(True)

    # Silhouette plot
    ax[1].plot(ks, silhouette_scores, marker='o', color='green')
    ax[1].set_xlabel('Number of clusters (k)')
    ax[1].set_ylabel('Silhouette Score')
    ax[1].set_title('Silhouette Scores by k')
    ax[1].grid(True)

    st.pyplot(fig)


        # Train model and get cluster labels
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(X)
    silhouette = silhouette_score(X, labels)

# ================================
# HIERARCHICAL CLUSTERING
# ================================
elif method == "Hierarchical Clustering":
    st.markdown("## 🌲 Hierarchical Clustering")
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

    # Feature Distribution / EDA
    st.markdown("#### 🔍 Feature Distributions (Histogram)")
    st.markdown("""
    This section gives a quick visual overview of the **distribution** of each numeric feature.
    Histograms help us check for **skewed features**, **outliers**, or **non-normality** — all of which may impact clustering.
    """)
    features_df = df[selected_features]
    fig, ax = plt.subplots(figsize=(12, 8))
    features_df.hist(ax=ax, edgecolor="k", bins=15)
    plt.suptitle("Distribution of Selected Features", y=1.02)
    plt.tight_layout()
    st.pyplot(fig)

    # Standardize before clustering
    X_scaled = StandardScaler().fit_transform(features_df)

    # Dendrogram
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

    from scipy.cluster.hierarchy import linkage, dendrogram

    # Sidebar dropdown to allow the user to select a linkage method
    # Each method determines how distances between clusters are calculated
    linkage_method = st.sidebar.selectbox("Linkage Method", ["ward", "complete", "average", "single"], index=0,
        help="""
    - **ward**: minimizes within-cluster variance (best for Euclidean).
    - **complete**: uses the maximum distance between cluster points.
    - **average**: average distance between all pairs.
    - **single**: nearest neighbor (may cause chaining).
    """)

    # Compute the linkage matrix based on the selected method
    # This contains the hierarchical relationships between data points    
    Z = linkage(X_scaled, method=linkage_method)
    
    # Extract labels for each observation — useful for dendrogram leaf labeling    
    labels = df.index.astype(str).tolist()

    # Plot the dendrogram
    plt.figure(figsize=(20, 7))
    dendrogram(Z, labels=labels)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Data Point")
    plt.ylabel("Distance")
    st.pyplot(plt)

    # Choose number of clusters
    k = st.sidebar.slider("Number of Clusters", 2, 10, 3)
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
    cluster_labels = model.fit_predict(X_scaled)

    # Silhouette score
    silhouette = silhouette_score(X_scaled, cluster_labels)
    st.subheader("🔗 Hierarchical Clustering Results")
    st.write(f"Silhouette Score: {silhouette:.3f}")

    # Visualize clusters using PCA
    plot_pca_clusters(X_scaled, cluster_labels, "Cluster Visualization (PCA Reduced)")

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

    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Explained variance ratios
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # 1. Display variance explanation metrics
    st.subheader("📉 PCA Results")
    st.markdown("""
    **Explained Variance Ratio** tells us how much information (variance) each principal component captures.  
    **Cumulative Explained Variance** helps us decide how many components are needed to capture most of the dataset's structure.
    """)
    st.write("Explained Variance Ratio:", explained_variance.round(3))
    st.write("Cumulative Explained Variance:", cumulative_variance.round(3))

    # 2. PCA Scatter Plot
    st.markdown("#### 🔍 PCA Scatter Plot")
    st.markdown("""
    Shows data projected into the new coordinate system defined by the first two principal components.  
    Each point represents a sample. If the data forms visible clusters, PCA has preserved some structure.
    """)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, edgecolor='k')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA: 2D Projection of Data")
    plt.grid(True)
    st.pyplot(plt)

    # 3. Biplot (with optional feature names)
    if st.checkbox("Show PCA Biplot (with feature loadings)"):
        st.markdown("#### 🧭 Biplot: PCA Scores + Loadings")
        st.markdown("""
        A biplot overlays feature directions on the PCA plot.  
        Arrows show how original features contribute to principal components.  
        Longer arrows = stronger influence. Directions indicate correlation with components.
        """)
        loadings = pca.components_.T
        scaling_factor = 3.0  # Adjust this as needed
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, edgecolor='k')
        for i, feature in enumerate(selected_features):
            plt.arrow(0, 0, scaling_factor * loadings[i, 0], scaling_factor * loadings[i, 1],
                    color='r', width=0.01, head_width=0.05)
            plt.text(scaling_factor * loadings[i, 0] * 1.1, scaling_factor * loadings[i, 1] * 1.1,
                    feature, color='r', ha='center', va='center')
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("PCA Biplot")
        plt.grid(True)
        st.pyplot(plt)

    # 4. Scree Plot
    st.markdown("#### 📊 Scree Plot: Cumulative Explained Variance")
    st.markdown("""
    This plot helps you determine how many components to retain by showing cumulative variance.  
    Look for the **elbow point** where adding more components yields diminishing returns.
    """)
    pca_full = PCA(n_components=X.shape[1])
    X_pca_full = pca_full.fit_transform(X)
    cumulative_variance_full = np.cumsum(pca_full.explained_variance_ratio_)
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(cumulative_variance_full)+1), cumulative_variance_full, marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Scree Plot")
    plt.grid(True)
    st.pyplot(plt)

    # 5. Bar Plot of Individual Variance
    st.markdown("#### 📌 Bar Plot: Variance Explained by Each Component")
    st.markdown("""
    Each bar shows how much variance is explained by a single principal component.  
    Useful for identifying the most important components.
    """)
    plt.figure(figsize=(8, 6))
    components = np.arange(1, len(pca_full.explained_variance_ratio_) + 1)
    plt.bar(components, pca_full.explained_variance_ratio_, alpha=0.7, color='teal')
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained")
    plt.title("Variance by Principal Component")
    plt.grid(True, axis='y')
    st.pyplot(plt)

    # 6. Combined Plot (Bar + Line)
    st.markdown("#### 📈 Combined Plot: Individual and Cumulative Variance")
    st.markdown("""
    This plot combines:
    - Bars for variance explained by each component
    - Line for cumulative variance

    **Interpretation:**  
    Look for the minimum number of components that explain ~90% of the variance.
    """)
    explained = pca_full.explained_variance_ratio_ * 100
    cumulative = np.cumsum(explained)

    fig, ax1 = plt.subplots(figsize=(8, 6))
    bar_color = 'steelblue'
    line_color = 'crimson'

    # Bar plot
    ax1.bar(components, explained, color=bar_color, alpha=0.8, label="Individual Variance")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained (%)", color=bar_color)
    ax1.tick_params(axis='y', labelcolor=bar_color)
    ax1.set_xticks(components)

    # Add % labels
    for i, v in enumerate(explained):
        ax1.text(components[i], v + 1, f"{v:.1f}%", ha='center', va='bottom', fontsize=10)

    # Line plot
    ax2 = ax1.twinx()
    ax2.plot(components, cumulative, color=line_color, marker='o', label="Cumulative Variance")
    ax2.set_ylabel("Cumulative Variance (%)", color=line_color)
    ax2.tick_params(axis='y', labelcolor=line_color)
    ax2.set_ylim(0, 100)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", bbox_to_anchor=(0.85, 0.5))

    plt.title("PCA: Variance Explained")
    plt.tight_layout()
    st.pyplot(fig)
