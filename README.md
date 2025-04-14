# Goulding-Data-Science-Portfolio

## Overview:

In this portfolio, you’ll find a variety of projects that showcase my skills in Python and machine learning. From building machine learning models to creating interactive data visualizations, each project illustrates my ability to turn complex data into actionable insights.

## Organization:

A summary of all projects can be seen below:

-    Penguin Filter Hub
-    Mutant Moneyball Cleaning and Analysis
-    Supervised Machine Learning Explorer

----------------------------------------------------------------------
### Penguin Filter Hub

[Click here!](https://github.com/ashleygoulding/Goulding-Data-Science-Portfolio/blob/main/basic-streamlit-app/main.py)

#### The Problem

Exploring large datasets can be overwhelming, especially when users need to filter and analyze specific subsets of data. The penguin dataset includes multiple attributes such as species, body mass, flipper length, and island location. Without an interactive interface, identifying trends and making meaningful observations is challenging.

#### The Solution

To address this, I built an interactive Streamlit web app that allows users to filter and explore the dataset effortlessly. Key steps included:
- *Building a user-friendly UI*: Using Streamlit widgets (`slider`, `selectbox`) to enable real-time filtering.
- *Filtering the dataset dynamically*: Implementing interactive filters for species, island, body mass, and flipper length.
- *Displaying real-time results*: Using `st.dataframe()` to update and show filtered data instantly.
- *Enhancing usability*: Providing a clean interface for intuitive exploration of penguin characteristics.

#### How This Complements My Portfolio

This project showcases my ability to:
- *Develop interactive data applications*: Creating real-time filtering tools using Streamlit.
- *Work with structured datasets*: Loading and presenting data effectively with pandas.
- *Enhance user experience*: Designing an intuitive UI that allows seamless data exploration.
- *Build functional, interactive tools*: Enabling users to analyze and visualize data without needing coding expertise.

----------------------------------------------------------------------
## Mutant Moneyball Cleaning and Analysis

[Click here!](https://github.com/ashleygoulding/Goulding-Data-Science-Portfolio/blob/main/TidyData-Project/TidyData.ipynb)

#### The Problem

Raw data is often messy, inconsistent, and difficult to analyze. In this project, the dataset from Rally's Mutant Moneyball article contained values that were poorly formatted, contained unnecessary characters, and were spread across multiple columns in an untidy structure. Without proper organization, drawing meaningful insights and creating visualizations was challenging.

#### The Solution

To solve this, I applied Tidy Data Principles to clean, transform, and restructure the dataset. Key steps included:
- *Cleaning the data*: Removing unwanted characters, handling missing values, and converting data types for consistency.
- *Reshaping the dataset*: Using `pd.melt()` to convert wide-format data into a long, tidy format.
- *Summarizing insights*: Creating pivot tables with `pd.pivot()` to analyze trends by source and decade.
- *Visualizing key findings*: Generating charts to highlight patterns and trends effectively.

![image](https://github.com/user-attachments/assets/70c16f4f-eaaa-4a06-97ea-be87278704aa)
![image](https://github.com/user-attachments/assets/72c176b3-bb5a-421a-a930-a200ae0c250f)

### How this Complements My Portfolio

This project demonstrates my ability to:
- Work with real-world data: Cleaning and structuring datasets efficiently.
- Apply Python data science libraries: Using `pandas`, `numpy`, and `matplotlib` for data manipulation and visualization.
- Follow best practices in data analysis: Ensuring reproducibility, clarity, and efficiency in code.
- Tell a story with data: Visualizing key insights to make complex information more understandable.

----------------------------------------------------------------------

### Supervised Machine Learning Explorer

[Click Here!](https://github.com/ashleygoulding/Goulding-Data-Science-Portfolio/tree/main/MLStreamlitApp)

#### The Problem

Many machine learning applications require users to understand complex models and their associated performance metrics. For individuals new to machine learning or those working with multiple datasets, choosing the right model, setting hyperparameters, and interpreting results can be daunting. There’s a need for an intuitive interface that simplifies the process of training models, evaluating their performance, and visualizing key metrics.

#### The Solution

To address this challenge, I developed an interactive app using Streamlit that allows users to explore supervised machine learning models and evaluate their performance. The app includes:
- **Data Upload**: Users can upload their own dataset or use a sample dataset to get started.
- **Model Selection**: Users can choose from different classifiers like Logistic Regression, Decision Trees, or K-Nearest Neighbors.
- **Hyperparameter Tuning**: The app provides sliders for tuning hyperparameters such as regularization strength, maximum depth, and number of neighbors.
- **Performance Metrics**: The app displays key metrics like accuracy, precision, recall, confusion matrix, and ROC curves to evaluate the model.
- **Model Visualization**: For decision trees, the app generates a visual representation of the tree to help users understand how decisions are made.

<img width="525" alt="image" src="https://github.com/user-attachments/assets/52dbbbd3-a749-42b4-a2b7-26fab241a313" />
<img width="525" alt="image" src="https://github.com/user-attachments/assets/678f57b1-4ecb-4290-bb0f-e5deaa771a1d" />

#### How this Complements my Portfolio
This project demonstrates my ability to:
- **Build interactive applications**: Using Streamlit, I created an app that provides real-time feedback and allows users to interact with machine learning models.
- **Work with machine learning models**: I’ve applied machine learning algorithms like Logistic Regression, Decision Trees, and KNN and used standard preprocessing techniques such as feature scaling and encoding.
- **Evaluate model performance**: I integrated performance metrics, including confusion matrices, classification reports, and ROC curves, to help users interpret model results.
- **Deploy Python solutions**: The app is a fully deployable solution, showcasing my skills in integrating machine learning models into interactive web applications.

----------------------------------------------------------------------
