import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import missingno as msno
#https://streamlit.io

# Load spreadsheet with studies
file_path = "clean_PRECISE_annotations.xlsx"
new_df = pd.read_excel(file_path, sheet_name='Sheet1')

# Streamlit Dashboard Setup
st.title("Dataset Dashboard")

# Basic Summary
st.subheader("Dataset Summary")
st.write({
    "Total Rows": len(new_df),
    "Total Columns": len(new_df.columns),
    "Missing Values": new_df.isnull().sum().sum(),
    "Columns with Missing Values": new_df.isnull().sum()[new_df.isnull().sum() > 0].to_dict()
})

# Count values for key categorical columns
df = new_df.replace(('No weight reported', 'No age reported', 'No sex reported', 'No strain reported', 'No species reported'), value=None)


new_columns = ["metadata:sex", "metadata:species", "metadata:tbi_model", "metadata:tbi_device:type", "metadata:age:category", "min_weight", "max_weight", "units_weight", "min_weeks", "max_weeks", "PMID", "metadata:strain", "metadata:tbi_device", "metadata:tbi_model_class", "metadata:tbi_device:angle (degrees from vertical)", "metadata:tbi_device:craniectomy_size", "metadata:tbi_device:dural_tears", "metadata:tbi_device:impact_area", "metadata:tbi_device:impact_depth (mm)", "metadata:tbi_device:impact_duration (ms)", "metadata:tbi_device:impact_velocity (m/s)", "metadata:tbi_device:shape"]

df_filt = df[new_columns]

categorical_columns =  ["metadata:sex", "metadata:species", "metadata:strain","metadata:tbi_model_class"]

#need to add more columns
'''for col in categorical_columns:
    if col in df_filt.columns:
        st.subheader(f"Distribution of {col} by TBI class")
        fig, ax = plt.subplots()
        sns.countplot(y=df_filt, x=col, hue="metadata:tbi_model_class", ax=ax)
        st.pyplot(fig)
'''

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


#Missing data analysis - all 

df.replace(r'^\s*$', np.nan, regex=True)
fig,ax = plt.subplots(figsize=(10,5))
msno.matrix(new_df, ax=ax, fontsize=12, color= (0.93, 0.00, 0.37), sparkline=False)

red_patch = mpatches.Patch(color= (0.93, 0.00, 0.37), label='Data present')
white_patch = mpatches.Patch(color='white', label='Data absent')

st.legend(handles=[red_patch, white_patch],loc='center left', bbox_to_anchor=(1.2, 0.7))

st.pyplot(fig)


#Missing data analysis - CCI
cci_df = df_filt[df_filt["metadata:tbi_model_class"]== "Controlled cortical impact model"]

fig,ax = plt.subplots(figsize=(10,5))
msno.matrix(cci_df, ax=ax, fontsize=12, color= (0.93, 0.00, 0.37), sparkline=False)

red_patch = mpatches.Patch(color= (0.93, 0.00, 0.37), label='Data present')
white_patch = mpatches.Patch(color='white', label='Data absent')

st.legend(handles=[red_patch, white_patch],loc='center left', bbox_to_anchor=(1.2, 0.7))

st.pyplot(fig)
