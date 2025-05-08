# 🐧 Penguin Data Explorer

The Penguin Filter Hub is a lightweight and interactive Streamlit web app designed to explore the Palmer Penguins dataset. It enables users to filter penguins by species, island, flipper length, and body mass in real time. The goal is to create an intuitive data exploration tool for both new and experienced data enthusiasts, without requiring any coding.

## 🚀 How to Run the App

### 🔧 Requirements

Make sure you have the following installed:

- Python 3.8+
- streamlit
- pandas

Install dependencies using pip:

```bash
pip install streamlit pandas 
```

### ▶️ Running the App Locally

Open your terminal or command prompt, navigate to the directory containing your Python script (e.g., `your_app_name.py`), and run the Streamlit app using the following command:

```bash
streamlit run basic-streamlit-app\main.py
```
### 🌐 Accessing it Online

Additionally, you can also access the app by using this [link](http://localhost:8501/) to go straight to the app on your browser

---

## 🚀 App Features
This app includes a series of filters that dynamically update the dataset view:
- **Species Filter**: Select a species (Adelie, Chinstrap, or Gentoo) to view relevant entries.
- **Island Filter**: Explore penguins from different islands (Biscoe, Dream, or Torgersen).
- **Body Mass Slider**: Adjust the slider to display penguins with body mass less than or equal to your selected value.
- **Flipper Length Slider**: Similar to body mass, this filter allows users to inspect penguins based on flipper length.

Each filter is interactive and updates the dataset view live, enabling intuitive exploration and quick pattern recognition.

---

## 📊 Visual Examples
Here are a few example screenshots from the app:

### 🐧 Species Filter
<img width="500" alt="species-filter" src="https://github.com/user-attachments/assets/569013d7-04e9-48b0-81ac-c2c9ebaa3625">

### 🏝️ Island Filter
<img width="500" alt="island-filter" src="https://github.com/user-attachments/assets/7c1fb58c-5567-414d-88de-8b1b64405c62">

---

## References
- [Palmer Penguins Dataset](https://allisonhorst.github.io/palmerpenguins/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
