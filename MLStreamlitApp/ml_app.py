import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay, roc_curve, auc, classification_report
from sklearn import tree
import graphviz

# Load example dataset
@st.cache_data
def load_data():
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame
    df['target'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df

st.set_page_config(page_title="ML Model Explorer", layout="wide")
st.title("Supervised Machine Learning Explorer")

st.markdown("""
Welcome to the **Supervised Machine Learning Explorer**!  
Upload your dataset, choose a model, adjust hyperparameters, and visualize results — all in one place.
""")

# Allow User to upload their own dataset
st.subheader("1️⃣ Upload a Dataset")
st.markdown("Upload a CSV file to get started. If none is uploaded, the Iris dataset will be used.")

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = load_data()

st.subheader("2️⃣ Dataset Preview")
st.dataframe(df.head())

st.subheader("3️⃣ Select Target and Features")
st.markdown("Pick your **target column** (what you want to predict) and the **features** (variables used for prediction).")
label_column = st.selectbox("Select target column:", options=df.select_dtypes(include=['object', 'category']).columns)
feature_columns = st.multiselect("Select features:", options=[col for col in df.columns if col != label_column], default=[col for col in df.columns if col != label_column])

if not feature_columns:
    st.warning("Please select at least one feature column.")
    st.stop()

# Select raw features and target
X_raw = df[feature_columns]
y_raw = df[label_column]

# Drop rows where X or y has missing values
df_clean = pd.concat([X_raw, y_raw], axis=1).dropna()
X_raw = df_clean[feature_columns]
y_raw = df_clean[label_column]

# Encode features
X = pd.get_dummies(X_raw)

# Encode target
if y_raw.dtype == 'object' or isinstance(y_raw.dtype, pd.CategoricalDtype):
    y = LabelEncoder().fit_transform(y_raw)
else:
    y = y_raw

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.sidebar.markdown("Select a machine learning model and customize hyperparameters below.")
st.sidebar.subheader("Choose Model")
model_type = st.sidebar.selectbox("Classifier:", ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors"])

# Define models
st.subheader("4️⃣ Model")

if model_type == "Logistic Regression":
    C = st.sidebar.slider("C (Inverse Regularization Strength)", 0.01, 10.0, 1.0)
    model = LogisticRegression(C=C, max_iter=200)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] if len(np.unique(y)) == 2 else None

    st.subheader("Logistic Regression Coefficients")
    coefs = pd.Series(model.coef_[0], index=X.columns)
    st.write(coefs.sort_values(ascending=False).to_frame("Coefficient"))
    
elif model_type == "Decision Tree":
    max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if len(np.unique(y)) == 2 else None

    st.subheader("Decision Tree Visualization")
    dot_data = export_graphviz(model, feature_names=X.columns, class_names=[str(c) for c in np.unique(y)], filled=True)
    graph = graphviz.Source(dot_data)
    st.graphviz_chart(dot_data)

elif model_type == "K-Nearest Neighbors":
    n_neighbors = st.sidebar.slider("Number of Neighbors (k)", 1, 15, 5)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] if len(np.unique(y)) == 2 else None

    st.subheader("KNN: Explore Effect of Different k Values")
    k_values = range(1, 21, 2)
    accuracies_scaled = []
    for k in k_values:
        knn_temp = KNeighborsClassifier(n_neighbors=k)
        knn_temp.fit(X_train_scaled, y_train)
        y_temp_pred = knn_temp.predict(X_test_scaled)
        accuracies_scaled.append(accuracy_score(y_test, y_temp_pred))

    fig_k = plt.figure(figsize=(8, 5))
    plt.plot(k_values, accuracies_scaled, marker='o')
    plt.title('Accuracy vs. Number of Neighbors (k) on Scaled Data')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.xticks(k_values)
    st.pyplot(fig_k)

# Metrics
st.subheader("Model Evaluation Metrics")
st.markdown("Evaluate your model’s performance using accuracy, precision, and recall.")
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
st.metric("Accuracy", f"{acc:.2%}")
st.metric("Precision", f"{precision:.2f}")
st.metric("Recall", f"{recall:.2f}")

# Confusion Matrix
st.subheader("Confusion Matrix")
st.markdown("This matrix shows how often predictions match actual labels. Great for spotting model strengths and weaknesses.")
fig_cm, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(model, X_test_scaled if model_type != "Decision Tree" else X_test, y_test, ax=ax, cmap='Blues')
st.pyplot(fig_cm)

# Classification Report
st.subheader("Detailed Classification Report")
st.markdown("See precision, recall, and F1-score for each class.")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.style.format("{:.2f}"))

# ROC Curve
# Check if binary classification and model supports predict_proba
st.markdown("### 📈 ROC Curve (Binary Classifiers Only)")
st.markdown("Visualize the trade-off between true positive and false positive rates.")

if len(np.unique(y)) == 2 and hasattr(model, "predict_proba"):
    # Compute predicted probabilities for the positive class
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2,
                label='ROC curve (area = %0.2f)' % roc_auc)
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Characteristic')
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)
else:
    st.warning("ROC curve only works for binary classifiers that support probability predictions.")

st.sidebar.markdown("---")
st.sidebar.markdown("💡 *Tip: Experiment with models and hyperparameters to improve performance!*")