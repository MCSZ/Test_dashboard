import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
#https://streamlit.io

# Load spreadsheet with studies
file_path = "clean_PRECISE_annotations.xlsx"
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Streamlit Dashboard Setup
st.title("Dataset Dashboard")

# Basic Summary
st.subheader("Dataset Summary")
st.write({
    "Total Rows": len(df),
    "Total Columns": len(df.columns),
    "Missing Values": df.isnull().sum().sum(),
    "Columns with Missing Values": df.isnull().sum()[df.isnull().sum() > 0].to_dict()
})

# Count values for key categorical columns
categorical_columns = ["metadata:assessment", "metadata:sex", "metadata:species", "metadata:tbi_model", "metadata:tbi_device:type"]
#need to add more columns
for col in categorical_columns:
    if col in df.columns:
        st.subheader(f"Distribution of {col}")
        fig, ax = plt.subplots()
        sns.countplot(y=df[col], order=df[col].value_counts().index, ax=ax)
        st.pyplot(fig)

# Data Summary
st.subheader("Numeric Data Summary")
numeric_columns = df.select_dtypes(include=["number"]).columns
st.write(df[numeric_columns].describe())

# Plot histograms for numerical columns
for col in numeric_columns:
    st.subheader(f"Distribution of {col}")
    fig, ax = plt.subplots()
    sns.histplot(df[col].dropna(), bins=20, kde=True, ax=ax)
    st.pyplot(fig)
