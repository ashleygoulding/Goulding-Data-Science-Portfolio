# 📊 Tidy Data Project

## 📌 Project Overview
This project applies **Tidy Data Principles** to transform and analyze a dataset, ensuring that:
1. 📌 Each variable forms a column.
2. 📌 Each observation forms a row.
3. 📌 Each type of observational unit forms its own table.

By following these principles, we make data manipulation, visualization, and analysis more efficient. The project includes data cleaning, restructuring, and visualization using **pandas** and **matplotlib**.

## ⚙️ Instructions
### 🔧 Prerequisites
Ensure you have Python installed along with the following dependencies:
```bash
pip install pandas numpy matplotlib seaborn
```
# 🚀 Running the Notebook

1. Open the Jupyter Notebook environment:

```bash
jupyter notebook
```
2. Navigate to the project directory and open TidyData.ipynb.

3. Run the notebook cells sequentially to:

- ✨ Clean and preprocess the data.
- 🔄 Reshape the dataset into a tidy format.
- 📊 Generate visualizations to explore trends

# 📂 Dataset Description

- 📖 Source: The dataset contains structured information on valuations across different decades, sources, and members.

- 🔍 Preprocessing Steps:
  - 🧹 Removing special characters from numerical values.
  - 🔄 Converting data into a structured, long format using pd.melt().
  - 📊 Creating pivot tables to summarize key trends.

🔗 References

For further reading on tidy data principles and data manipulation techniques, see:

📄 Pandas Cheat Sheet

📑 Tidy Data Paper by Hadley Wickham

📈 Visual Examples

Below are examples of the visualizations generated in this project:

📊 Total Value Trends Over Decades (Line Chart)

📉 Source Contribution Over Time (Stacked Bar Chart)

📦 Valuation Distribution by Source (Boxplot)

🔥 Heatmaps of Member Valuations by Source and Decade

For more details, run the notebook and explore the results interactively! 🎯
