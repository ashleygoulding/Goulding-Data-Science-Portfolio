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

### ▶️ Running the App Locally

Open your terminal or command prompt, navigate to the directory containing your Python script (e.g., `your_app_name.py`), and run the Streamlit app using the following command:

```bash
streamlit run MLStreamlitApp\ml_app.py
```
### 🌐 Accessing it Online

Additionally, you can also access the app by using this [link](https://goulding-data-science-portfolio-ma7wqhuxjsyvzceqycydvu.streamlit.app/) to go straight to the app on your browser

---

## ⚙️ App Features

1. **Data Upload and Preview**
- Users can upload their own CSV files.
- If no file is uploaded, the Iris dataset is loaded as an example.
- The app displays the first few rows of the loaded dataset for a quick preview.

2. **Feature and Target Selection**
- Users can select a single column as the **target variable** (the variable to be predicted).
- Users can select multiple columns as **features** (the variables used for prediction).
- The app handles categorical target variables by using `LabelEncoder`.
- Categorical features are automatically handled using one-hot encoding (`pd.get_dummies`).
- Rows with missing values in the selected features or target are automatically dropped to ensure clean training data.

3. **Machine Learning Models**
The app currently supports the following classification models:

    -   **Logistic Regression:**
        -   Users can adjust the `C` hyperparameter (inverse of regularization strength) using a slider.
        -   The app displays the learned coefficients of the Logistic Regression model.

    -   **Decision Tree:**
        -   Users can control the `max_depth` hyperparameter of the tree using a slider.
        -   The app visualizes the trained Decision Tree using `graphviz`.

    -   **K-Nearest Neighbors (KNN):**
        -   Users can set the `n_neighbors` hyperparameter (the number of neighbors to consider) using a slider.
        -   The app includes a plot showing the accuracy of the KNN model for different values of `k` (from 1 to 20 with a step of 2) on the scaled test data, helping users understand the impact of this crucial hyperparameter.

4.  **Model Evaluation Metrics:**
    After training and making predictions, the app displays the following evaluation metrics:
    -   **Accuracy:** The overall proportion of correctly classified instances.
    -   **Precision:** The proportion of correctly predicted positive instances out of all instances predicted as positive (for each class, weighted average is shown for multiclass).
    -   **Recall:** The proportion of correctly predicted positive instances out of all actual positive instances (for each class, weighted average is shown for multiclass).

5.  **Confusion Matrix:**
    -   A visual representation of the model's performance, showing the counts of true positives, true negatives, false positives, and false negatives.

6.  **Classification Report:**
    -   A detailed report providing precision, recall, F1-score, and support for each class in the target variable.

7.  **ROC Curve (for Binary Classification):**
    -   For binary classification problems (where the target variable has only two unique values), the app displays the Receiver Operating Characteristic (ROC) curve.
    -   It also calculates and displays the Area Under the Curve (AUC), which summarizes the overall performance of the model across all classification thresholds.
    -   A warning is displayed if the target variable is not binary or if the selected model does not support probability predictions.

8.  **Hyperparameter Tuning:**
    -   Key hyperparameters for each selected model are exposed through interactive sliders in the sidebar, allowing users to easily experiment with different settings and observe their impact on model performance.

---

## 🖼️ Visual Examples

### Confusion Matrix
<img width="525" alt="image" src="https://github.com/user-attachments/assets/52dbbbd3-a749-42b4-a2b7-26fab241a313" />

### Decision Tree
<img width="525" alt="image" src="https://github.com/user-attachments/assets/2a9fed58-a51a-4feb-91f2-8b8fbe468faa" />

### ROC Curve
<img width="525" alt="image" src="https://github.com/user-attachments/assets/678f57b1-4ecb-4290-bb0f-e5deaa771a1d" />

---

## 📚 References

-   **Streamlit Documentation:** [https://streamlit.io/docs/](https://streamlit.io/docs/)
-   **Pandas Documentation:** [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
-   **NumPy Documentation:** [https://numpy.org/doc/stable/](https://numpy.org/doc/stable/)
-   **Scikit-learn Documentation:** [https://scikit-learn.org/stable/user_guide.html](https://scikit-learn.org/stable/user_guide.html)
-   **Seaborn Documentation:** [https://seaborn.pydata.org/api.html](https://seaborn.pydata.org/api.html)
-   **Matplotlib Documentation:** [https://matplotlib.org/stable/contents.html](https://matplotlib.org/stable/contents.html)
-   **Graphviz Documentation:** [https://graphviz.org/documentation/](https://graphviz.org/documentation/)
-   **Label Encoding in Scikit-learn:** [https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
-   **StandardScaler in Scikit-learn:** [https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
-   **Logistic Regression in Scikit-learn:** [https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
-   **Decision Trees in Scikit-learn:** [https://scikit-learn.org/stable/modules/tree.html](https://scikit-learn.org/stable/modules/tree.html)
-   **K-Nearest Neighbors in Scikit-learn:** [https://scikit-learn.org/stable/modules/neighbors.html#classification](https://scikit-learn.org/stable/modules/neighbors.html#classification)
-   **Model Evaluation in Scikit-learn:** [https://scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html)
