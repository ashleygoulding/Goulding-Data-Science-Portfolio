# 🧠 Supervised Machine Learning Explorer

A web-based interactive application built with Streamlit to explore and visualize supervised machine learning models. Upload your dataset, select target variables and features, tune model hyperparameters, and gain insights through detailed metrics and visualizations — all from your browser!

---

## 📌 Project Overview

The Supervised Machine Learning Explorer provides an intuitive interface for beginners and intermediate users to:

-Load or upload datasets (default is the Iris dataset)
-Select features and target variable
-Apply and compare multiple classification models
-Visualize performance with confusion matrices, ROC curves, and classification reports

Whether you're learning ML or building prototypes, this tool helps make model experimentation fast, visual, and interactive.

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
- graphviz

Install dependencies using pip:

```bash
pip install streamlit pandas numpy seaborn matplotlib scikit-learn graphviz
```

### ▶️ Running the App

To run the app, paste the following code in the terminal:

```bash
streamlit run MLStreamlitApp\ml_app.py
```
### 🌐 Try it Online

Additionally, you can also access the app by using this [link](https://goulding-data-science-portfolio-ma7wqhuxjsyvzceqycydvu.streamlit.app/) to go straight to the app on your browser

---

## ⚙️ App Features

### 1️⃣ Upload a Dataset
Upload your own `.csv` file or use the default Iris dataset.

### 2️⃣ Dataset Preview
View the top rows of your dataset in a clean table.

### 3️⃣ Select Target and Features
Choose which column to predict and which features to use.

### 4️⃣ Model Selection
Choose from three classifiers:
- **Logistic Regression**
- **Decision Tree**
- **K-Nearest Neighbors**
  
Each model includes interactive widgets in the sidebar to customize hyperparameters like:
- `C` for regularization strength (Logistic Regression)
- `max_depth` (Decision Tree)
- `k` neighbors (KNN)

### 📊 Model Evaluation
Get real-time feedback on model performance:
- Accuracy, precision, recall
- Confusion matrix
- ROC curve (for binary classifiers)
- Full classification report

### 🌲 Decision Tree Visualization
<img width="517" alt="image" src="https://github.com/user-attachments/assets/6ee87c73-9087-42ef-998b-4d3e3d61dcd4" />
