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

## App Features

Explore the power of machine learning with these intuitive features:

<br>

<div style="display: flex; flex-wrap: wrap; gap: 20px;">

    <div style="flex: 1 1 300px; border: 1px solid #ddd; padding: 15px; border-radius: 8px; box-shadow: 2px 2px 5px #eee;">
        <h4 style="color: #336699; margin-top: 0;">📤 Dataset Upload & Preview</h4>
        <p style="margin-bottom: 10px;">
            Effortlessly upload your own CSV datasets to unlock personalized analysis. If you're just exploring, the classic Iris dataset is ready to go! Get a quick glimpse of your data with the integrated preview.
        </p>
        <ul style="list-style-type: disc; margin-left: 20px;">
            <li>Supports CSV file uploads.</li>
            <li>Loads the Iris dataset as a default.</li>
            <li>Displays a clear preview of the first few rows.</li>
        </ul>
    </div>

    <div style="flex: 1 1 300px; border: 1px solid #ddd; padding: 15px; border-radius: 8px; box-shadow: 2px 2px 5px #eee;">
        <h4 style="color: #336699; margin-top: 0;">🎯 Feature & Target Selection</h4>
        <p style="margin-bottom: 10px;">
            Clearly define your prediction goals by selecting a single target variable and the relevant features that will drive your model. The app intelligently handles different data types.
        </p>
        <ul style="list-style-type: disc; margin-left: 20px;">
            <li>Intuitive selection of the target column.</li>
            <li>Easy multi-select for feature columns.</li>
            <li>Automatic handling of categorical target variables (Label Encoding).</li>
            <li>Automatic handling of categorical features (One-Hot Encoding).</li>
            <li>Intelligent removal of rows with missing data.</li>
        </ul>
    </div>

</div>

<br>

<h3 style="color: #336699;">🤖 Explore Powerful Classification Models</h3>
<p>Dive into the world of supervised learning with these carefully integrated models:</p>

<div style="display: flex; flex-wrap: wrap; gap: 20px; margin-top: 15px;">

    <div style="flex: 1 1 300px; border: 1px solid #ddd; padding: 15px; border-radius: 8px; box-shadow: 2px 2px 5px #eee;">
        <h4 style="color: #336699; margin-top: 0;"><span style="font-size: 1.2em;">📈</span> Logistic Regression</h4>
        <p style="margin-bottom: 10px;">
            A fundamental algorithm for binary and multiclass classification. Adjust the regularization strength (`C`) to control model complexity and prevent overfitting.
        </p>
        <ul style="list-style-type: disc; margin-left: 20px;">
            <li>Interactive slider for the `C` hyperparameter.</li>
            <li>Clear display of the learned model coefficients.</li>
        </ul>
    </div>

    <div style="flex: 1 1 300px; border: 1px solid #ddd; padding: 15px; border-radius: 8px; box-shadow: 2px 2px 5px #eee;">
        <h4 style="color: #336699; margin-top: 0;"><span style="font-size: 1.2em;">🌳</span> Decision Tree</h4>
        <p style="margin-bottom: 10px;">
            A tree-based model that makes decisions based on a series of rules. Control the tree's depth (`max_depth`) to balance model complexity and interpretability.
        </p>
        <ul style="list-style-type: disc; margin-left: 20px;">
            <li>Interactive slider for the `max_depth` hyperparameter.</li>
            <li>Visual representation of the decision tree structure.</li>
        </ul>
    </div>

    <div style="flex: 1 1 300px; border: 1px solid #ddd; padding: 15px; border-radius: 8px; box-shadow: 2px 2px 5px #eee;">
        <h4 style="color: #336699; margin-top: 0;"><span style="font-size: 1.2em;"><0xE2><0x9C><0x8E></span> K-Nearest Neighbors (KNN)</h4>
        <p style="margin-bottom: 10px;">
            A simple yet effective algorithm that classifies data points based on the majority class among their `k` nearest neighbors. Experiment with different values of `k`.
        </p>
        <ul style="list-style-type: disc; margin-left: 20px;">
            <li>Interactive slider for the number of neighbors (`n_neighbors`).</li>
            <li>Visual exploration of accuracy across different `k` values.</li>
        </ul>
    </div>

</div>

<br>

<h3 style="color: #336699;">📊 Comprehensive Model Evaluation</h3>
<p>Gain deep insights into your model's performance with these detailed metrics and visualizations:</p>

<div style="display: flex; flex-wrap: wrap; gap: 20px; margin-top: 15px;">

    <div style="flex: 1 1 300px; border: 1px solid #ddd; padding: 15px; border-radius: 8px; box-shadow: 2px 2px 5px #eee;">
        <h4 style="color: #336699; margin-top: 0;">✅ Key Performance Metrics</h4>
        <ul style="list-style-type: disc; margin-left: 20px;">
            <li>**Accuracy:** Overall correctness of the model.</li>
            <li>**Precision:** Ability of the model to avoid false positives.</li>
            <li>**Recall:** Ability of the model to identify all relevant instances.</li>
            <li>Clear display of each metric for immediate understanding.</li>
        </ul>
    </div>

    <div style="flex: 1 1 300px; border: 1px solid #ddd; padding: 15px; border-radius: 8px; box-shadow: 2px 2px 5px #eee;">
        <h4 style="color: #336699; margin-top: 0;"><span style="font-size: 1.2em;"><0xF0><0x9F><0x97><0x82></span> Confusion Matrix</h4>
        <p style="margin-bottom: 10px;">
            Visualize the distribution of true positives, true negatives, false positives, and false negatives to understand where your model excels and where it struggles.
        </p>
    </div>

    <div style="flex: 1 1 300px; border: 1px solid #ddd; padding: 15px; border-radius: 8px; box-shadow: 2px 2px 5px #eee;">
        <h4 style="color: #336699; margin-top: 0;"><span style="font-size: 1.2em;">📄</span> Detailed Classification Report</h4>
        <p style="margin-bottom: 10px;">
            Get a class-by-class breakdown of precision, recall, F1-score, and support, providing a granular view of your model's performance on each category.
        </p>
    </div>

    <div style="flex: 1 1 300px; border: 1px solid #ddd; padding: 15px; border-radius: 8px; box-shadow: 2px 2px 5px #eee;">
        <h4 style="color: #336699; margin-top: 0;"><span style="font-size: 1.2em;">📈</span> ROC Curve (Binary Classifiers)</h4>
        <p style="margin-bottom: 10px;">
            For binary classification tasks, the Receiver Operating Characteristic (ROC) curve visually represents the trade-off between the true positive rate and the false positive rate across different classification thresholds. The Area Under the Curve (AUC) summarizes the model's overall discriminatory power.
        </p>
    </div>

</div>

<br>

<h3 style="color: #336699;">⚙️ Interactive Hyperparameter Tuning</h3>
<p>Experiment and fine-tune your models effortlessly using intuitive sliders in the sidebar. Observe in real-time how different hyperparameter settings impact model performance.</p>

--

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
