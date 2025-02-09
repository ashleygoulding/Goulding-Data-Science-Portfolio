import streamlit as st
import pandas as pd

st.title("Penguin Filter Hub")
st.text("Dive into the world of penguins! Filter and explore data on species, islands, bill size, and body mass!")

# Display the imported dataset
df2 = pd.read_csv("basic-streamlit-app\data\penguins.csv")
st.dataframe(df2)

# filters penguins by body mass 
body_mass = st.slider("Choose a body mass range:", 
                   min_value = df2["body_mass_g"].min(),
                   max_value = df2["body_mass_g"].max())

# Filtering the DataFrame based on user selection
st.write(f"Body mass under {body_mass}:")
st.dataframe(df2[df2['body_mass_g'] <= body_mass])

# filters penguins by species 
species = st.selectbox("Select a species", df2["species"].unique())

# Filtering the DataFrame based on user selection
filtered_df = df2[df2["species"] == species]

# Display the filtered results
st.write(f"{species} penguins:")
st.dataframe(filtered_df)

# filters penguins by flipper length 
flipper_length = st.slider("Choose a fliper length range:", 
                   min_value = df2["flipper_length_mm"].min(),
                   max_value = df2["flipper_length_mm"].max())

# Filtering the DataFrame based on user selection
st.write(f"Flipper length under {flipper_length}:")
st.dataframe(df2[df2['flipper_length_mm'] <= flipper_length])

# Filtering the DataFrame based on user selection
st.write(f"Body mass under {body_mass}:")
st.dataframe(df2[df2['body_mass_g'] <= body_mass])

# filters penguins by island 
island = st.selectbox("Select an island", df2["island"].unique())

# Filtering the DataFrame based on user selection
filtered_df = df2[df2["island"] == island]

# Display the filtered results
st.write(f"{island} penguins:")
st.dataframe(filtered_df)