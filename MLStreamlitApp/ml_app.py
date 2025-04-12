import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

# Load default Iris dataset
@st.cache_data
def load_data():
    iris = load_iris(as_frame=True)
    df = iris.frame
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df

st.title("🌸 Supervised Machine Learning Explorer")
st.write("Upload your dataset, select a model, and experiment with hyperparameters to observe model training and performance.")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Feature and target selection
st.subheader("Model Settings")

# Let user select target column (only categorical or object type)
label_column = st.selectbox("Select the target (label) column:",
                            options=df.select_dtypes(include=['object', 'category']).columns)

# Let user select features (exclude label)
feature_columns = st.multiselect("Select feature columns:",
                                 options=[col for col in df.columns if col != label_column],
                                 default=[col for col in df.columns if col != label_column])

if not feature_columns:
    st.warning("Please select at least one feature column.")
    st.stop()

# Define features and target
X_raw = df[feature_columns]
y_raw = df[label_column]

# Encode non-numeric features
X = pd.get_dummies(X_raw)

# Encode target if needed
if y_raw.dtype == 'object' or isinstance(y_raw.dtype, pd.CategoricalDtype):
    y = LabelEncoder().fit_transform(y_raw)
else:
    y = y_raw

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection
st.sidebar.subheader("Model Hyperparameters")
model_type = st.sidebar.selectbox("Choose a classifier:", ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors"])

if model_type == "Logistic Regression":
    C = st.sidebar.slider("Inverse Regularization Strength (C)", 0.01, 10.0, 1.0)
    model = LogisticRegression(C=C, max_iter=200)
elif model_type == "Decision Tree":
    max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)
    model = DecisionTreeClassifier(max_depth=max_depth)
elif model_type == "K-Nearest Neighbors":
    n_neighbors = st.sidebar.slider("Number of Neighbors (k)", 1, 15, 5)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

# Train model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Output
st.subheader("Model Performance")
acc = accuracy_score(y_test, y_pred)
st.metric("Accuracy", f"{acc:.2%}")

# Show classification report
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Plot confusion matrix
st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
st.pyplot(fig)

# If binary classification, show ROC curve
if len(np.unique(y)) == 2:
    st.subheader("ROC Curve")
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('Receiver Operating Characteristic')
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

# Allow users to experiment with different parameters and models
st.sidebar.subheader("Experiment & Explore!")
st.sidebar.write("Adjust the hyperparameters on the left sidebar and observe how the model's performance changes.")
