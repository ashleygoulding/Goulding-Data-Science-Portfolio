# 📊 Tidy Data Project

## 📌 Project Overview
This project showcases the application of **Tidy Data Principles** to transform and analyze a dataset, ensuring:

1. **Each variable forms a column**: This makes it easier to manipulate and analyze individual variables.
2. **Each observation forms a row**: This represents each record in the dataset in a structured manner.
3. **Each type of observational unit forms its own table**: Ensuring clear and efficient storage of data across different categories.

By adhering to these principles, we ensure the data is well-organized, enabling more efficient data manipulation, analysis, and visualization. This project involves the use of **pandas** for data cleaning and restructuring, and **matplotlib** for creating insightful visualizations to uncover trends and patterns.

### ✨ Key Benefits:
- **Easier Analysis**: By organizing the data in a tidy format, it’s easier to apply various analytical techniques.
- **Effective Visualization**: Clean and tidy data ensures that visualizations are more meaningful and insightful.
- **Reproducibility**: A tidy dataset ensures consistency and allows others to replicate the analysis easily.

---

## ⚙️ Instructions

### 🔧 Prerequisites
To run the project, ensure you have Python installed, along with the following dependencies:

```bash
pip install pandas numpy matplotlib seaborn
```

# 🚀 Running the Notebook

1. Open the Jupyter Notebook environment:

```bash
jupyter notebook
```
2. Navigate to the project directory and open `TidyData.ipynb`.

3. Run the notebook cells sequentially to:

- ✨ Clean and preprocess the dataset by handling missing values, standardizing formats, and removing outliers.
- 🔄 Reshape the dataset into a tidy format using `pd.melt()` to ensure variables are aligned in columns and rows properly.
- 📊 Generate visualizations to explore trends such as total value trends, source contributions over time, and valuation distributions.
  
# 📂 Dataset Description

## 📖 Source: 
This data set is from Rally's [Mutant Moneyball](https://rallyrd.com/mutant-moneyball-a-data-driven-ultimate-x-men/) article which visualizes X-Men value data, era by era, from the X-Men's creation in 1963 up to 1993.

## 🔍 Steps Taken:
1. Cleaning: Special characters are removed from numerical values to ensure that the data is in a usable format.
2. Reshaping: We convert the data into a structured, long format using pd.melt() to ensure that all variables are in separate columns.
3. Summarization: Pivot tables are created to summarize key trends, such as the contribution of different sources over time.
4. Visualization: Various charts were created to depict the most and least valuable decades, sources, and members of the X-Men.
   
# 🔗 References

For further reading on tidy data principles and data manipulation techniques, see:

- 📄 [Pandas Cheat Sheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf) : A quick reference guide for using pandas functions effectively.
- 📑 [Tidy Data Paper by Hadley Wickham](https://vita.had.co.nz/papers/tidy-data.pdf) : A paper by Hadley Wickham that outlines the importance of tidy data principles.

# 📈 Visual Examples

Below are examples of the visualizations generated in this project:
- 📊 Total Value Trends Over Decades (Line Chart)
  
![image](https://github.com/user-attachments/assets/c06c9401-7869-4a18-a18d-5f56ba50b4ad)

- 📉 Source Contribution Over Time (Stacked Bar Chart)
  
![image](https://github.com/user-attachments/assets/6f98da3c-32b5-4369-a097-e78208e31a7d)

- 📦 Valuation Distribution by Source (Boxplot)
  
![image](https://github.com/user-attachments/assets/f4452157-f509-4e42-8148-844c346951a6)

