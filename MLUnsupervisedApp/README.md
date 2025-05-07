# 🧠 Unsupervised Learning Explorer

A user-friendly Streamlit app for exploring **unsupervised machine learning** techniques—**K-Means Clustering**, **Hierarchical Clustering**, and **Principal Component Analysis (PCA)**—with rich visualizations and interactive controls.

## 📌 Project Overview

This app helps users explore patterns and structures in datasets without predefined labels. Users can:

- Upload their own CSV datasets or select sample data.
- Choose features for clustering and dimensionality reduction.
- Run **K-Means** and **Hierarchical Clustering**, with visualizations like:
  - Elbow plots
  - Silhouette plots
  - Dendrograms
- Perform **PCA**, with explained variance ratios, 2D projections, and interactive biplots.

It’s ideal for students, analysts, or anyone interested in unsupervised learning without needing to write code.

---

## 🚀 How to Run the App

### 🔧 Requirements

Make sure you have the following installed:

- Python 3.8+
- streamlit
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

Install dependencies using pip:

```bash
pip install streamlit pandas numpy seaborn matplotlib scikit-learn 
```

### ▶️ Running the App Locally

Open your terminal or command prompt, navigate to the directory containing your Python script (e.g., `your_app_name.py`), and run the Streamlit app using the following command:

```bash
streamlit run MLUnsupervisedApp\ml_upsup_app.py
```
### 🌐 Accessing it Online

Additionally, you can also access the app by using this [link](https://goulding-data-science-portfolio-6ljc4fdj5tutg2pvmvruqg.streamlit.app/) to go straight to the app on your browser

---

## ⚙️ App Features

This app is divided into three core modules—K-Means Clustering, Hierarchical Clustering, and Principal Component Analysis (PCA)—each designed with both functionality and interpretability in mind. Below is a detailed overview of each:

### Data Upload and Preview
- Users can upload their own CSV files.
- If no file is uploaded, the Iris or Wine dataset can be loaded as an example.
- The app displays the first few rows of the loaded dataset for a quick preview.

### 🧩 K-Means Clustering
K-Means is a partitioning algorithm that groups data into k clusters by minimizing the variance within each cluster.
- **Cluster Range Selection**: Users can specify a custom range of cluster values (e.g., 2–10) to evaluate.
- **Elbow Plot**: Visualizes the *Sum of Squared Errors (SSE)* across different k values to help identify the "elbow point" where adding more clusters yields diminishing returns.
- **Silhouette Plot**: Plots the average silhouette score for each k, offering insight into the separation and cohesion of clusters.
- **Cluster Visualization**: Plots the clustering results in 2D using either raw data (if 2D) or PCA-reduced dimensions for higher-dimensional data.
- **Implementation**: Uses `KMeans` from `scikit-learn`, with default `k-means++` initialization for smarter centroid selection.

### 🌳 Hierarchical Clustering
Hierarchical clustering builds a tree (dendrogram) of nested clusters, allowing users to explore groupings at different levels of granularity.
- **Linkage Selection**: Supports multiple linkage methods: `ward` (default), `complete`, `average`, and `single`, allowing flexible control over how clusters are merged.
- **Dendrogram Generation**: Displays a dendrogram that visualizes the hierarchy of clusters and helps determine an appropriate number of clusters by inspecting linkage distances.
- **Cluster Selection**: After visual inspection, users can specify the number of clusters to cut the dendrogram and view resulting cluster labels.
- **2D Visualization**: If data is multidimensional, the app uses PCA to reduce it to 2D for intuitive cluster visualization.
- **Implementation**: Combines `AgglomerativeClustering` from `scikit-learn` and `dendrogram` generation from  `scipy.cluster.hierarchy`.

### 📉 Principal Component Analysis (PCA)
PCA reduces the dimensionality of data while preserving as much variance as possible, enabling clearer insights and improved visualizations.
- **Explained Variance Plot**: Shows the variance explained by each principal component, helping users decide how many components to retain.
- **2D PCA Projection**: Projects high-dimensional data onto the first two principal components, enabling cluster visualization and structure discovery.
- **Interactive Biplot *(optional)***: Overlays original feature vectors onto the 2D PCA projection, illustrating how features contribute to each principal component.
- **Component Control**: Users can choose how many components to calculate and whether to standardize the data before applying PCA.
- **Implementation**: Uses `PCA` from `scikit-learn`, with optional standardization using `StandardScaler`.

---
## 🖼️ Visual Examples

### Elbow and Silhouette Plot (K-Means)
![image](https://github.com/user-attachments/assets/2ec6a392-3730-4b73-8c6f-b180b42184fd)

### Dendrogram (Hierarchical Clustering)
![image](https://github.com/user-attachments/assets/4ada3de2-a041-40c4-a39f-eef5c1b426a8)

### PCA Biplot
![image](https://github.com/user-attachments/assets/cb1b4379-1994-49e8-ad87-51521f056869)

---

## 📚 References

- [Scikit-Learn Documentation](https://scikit-learn.org/stable/user_guide.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [PCA in Python – An Intuitive Guide](https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60)
- [Hierarchical Clustering Tutorial](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)
