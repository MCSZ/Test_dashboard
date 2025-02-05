import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import missingno as msno
import matplotlib.patches as mpatches
import numpy as np
#from sklearn.cluster import KMeans
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA

#https://streamlit.io

# Load spreadsheet with studies 
file_path = "clean_PRECISE_annotations.xlsx"
new_df = pd.read_excel(file_path, sheet_name='Sheet1')

# Streamlit Dashboard Setup
st.title("PRECISE-TBI Metadata Dashboard")

# Basic Summary
st.subheader("Metadata Summary")
new_columns = ["metadata:sex", "metadata:species", "metadata:tbi_model", "metadata:tbi_device:type", "metadata:age:category", "min_weight", "max_weight", "units_weight", "min_weeks", "max_weeks","metadata:strain", "metadata:tbi_device", "metadata:tbi_model_class", "metadata:tbi_device:angle (degrees from vertical)", "metadata:tbi_device:craniectomy_size", "metadata:tbi_device:dural_tears", "metadata:tbi_device:impact_area", "metadata:tbi_device:impact_depth (mm)", "metadata:tbi_device:impact_duration (ms)", "metadata:tbi_device:impact_velocity (m/s)", "metadata:tbi_device:shape"]

df_filt = new_df[new_columns]

st.write({
    "Total Rows": len(df_filt),
    "Total Columns": len(df_filt.columns),
    "Missing Values": df_filt.isnull().sum().sum(),
    "Columns with Missing Values": df_filt.isnull().sum()[df_filt.isnull().sum() > 0].to_dict()
})

# Replace some text in missing values to NAN
df = df_filt.replace(('No weight reported', 'No age reported', 'No sex reported', 'No strain reported', 'No species reported'), value=None)



categorical_columns =  ["metadata:sex", "metadata:species", "metadata:strain","metadata:tbi_model_class", "metadata:age:category"]

#need to add more columns
for col in categorical_columns:
    if col in df.columns:
        st.subheader(f"Distribution of {col}")
        fig, ax = plt.subplots()
        sns.countplot(data=df, y="metadata:tbi_model_class", hue=col, ax=ax)
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

# General Summary
st.subheader("General Summary")
general_summary = df.groupby('metadata:tbi_model_class').agg({
    'min_weeks': ['min', 'max'],
    'min_weight': ['min', 'max'],
    'metadata:species': pd.Series.nunique,
    'metadata:sex': pd.Series.nunique
}).reset_index()
general_summary.columns = ['TBI Model Class', 'Min Age (weeks)', 'Max Age (weeks)', 'Min Weight', 'Max Weight', 'Unique Species Count', 'Unique Sex Count']
st.write(general_summary)

#Missing data analysis - all 

df_filt.replace(r'^\s*$', regex=True)
fig,ax = plt.subplots(figsize=(10,5))
msno.matrix(df, ax=ax, fontsize=12, color= (0.93, 0.00, 0.37), sparkline=False)

red_patch = mpatches.Patch(color= (0.93, 0.00, 0.37), label='Data present')
white_patch = mpatches.Patch(color='white', label='Data absent')

ax.legend(handles=[red_patch, white_patch],loc='center left', bbox_to_anchor=(1.2, 0.7))

st.pyplot(fig)


# Age Classification Analysis
st.subheader("Age and Weight Distribution by Sex and Species")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x='min_weeks', y='min_weight', hue='metadata:sex', style='metadata:species', ax=ax)
ax.set_xlabel('Age (weeks)')
ax.set_ylabel('Weight')
st.pyplot(fig)

# Strain Analysis
st.subheader("Strain Classification Analysis")
df['strain_prefix'] = df['metadata:strain'].str[:4]
strain_summary = df.groupby(['strain_prefix', 'metadata:strain_class']).size().reset_index(name='Count')
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=strain_summary, y='strain_prefix', hue='metadata:strain_class', ax=ax)
ax.set_ylabel('Strain Prefix')
ax.set_xlabel('Count')
st.pyplot(fig)

# Model Summary
st.subheader("Model Summary by Age, Weight, Classification, Species, and Strain")
model_summary = df.groupby('metadata:tbi_model').agg({
    'min_weeks': ['min', 'max'],
    'min_weight': ['min', 'max'],
    'metadata:sex': pd.Series.nunique,
    'metadata:species': pd.Series.nunique,
    'metadata:strain': pd.Series.nunique
}).reset_index()
model_summary.columns = ['TBI Model', 'Min Age (weeks)', 'Max Age (weeks)', 'Min Weight', 'Max Weight', 'Unique Sex Count', 'Unique Species Count', 'Unique Strain Count']
st.write(model_summary)

# Controlled Cortical Impact Model Analysis
st.subheader("Controlled Cortical Impact Model: Missing Data Analysis on Injury Parameters")
cci_df = df[df['metadata:tbi_model_class'] == 'controlled cortical impact model']
injury_params = [
    "metadata:tbi_device:angle (degrees from vertical)",
    "metadata:tbi_device:craniectomy_size",
    "metadata:tbi_device:dural_tears",
    "metadata:tbi_device:impact_area",
    "metadata:tbi_device:impact_depth (mm)",
    "metadata:tbi_device:impact_duration (ms)",
    "metadata:tbi_device:impact_velocity (m/s)",
    "metadata:tbi_device:shape"
]
fig, ax = plt.subplots(figsize=(10, 5))
msno.matrix(cci_df[injury_params], ax=ax, fontsize=12, color=(0.93, 0.00, 0.37), sparkline=False)
st.pyplot(fig)

#Missing data analysis - CCI
st.subheader("CCI Model Papers - Missing Data Summary")
cci_col = ["metadata:tbi_model", "metadata:tbi_model_class", "metadata:tbi_device:type", "metadata:tbi_device", "metadata:tbi_device:angle (degrees from vertical)", "metadata:tbi_device:craniectomy_size", "metadata:tbi_device:dural_tears", "metadata:tbi_device:impact_area", "metadata:tbi_device:impact_depth (mm)", "metadata:tbi_device:impact_duration (ms)", "metadata:tbi_device:impact_velocity (m/s)", "metadata:tbi_device:shape"]
cci_newdf = new_df[cci_col]

cci_df = cci_newdf[cci_newdf["metadata:tbi_model_class"]== "Controlled cortical impact model"]

fig,ax = plt.subplots(figsize=(10,5))
msno.matrix(cci_df, ax=ax, fontsize=12, color= (0.93, 0.00, 0.37), sparkline=False)

red_patch = mpatches.Patch(color= (0.93, 0.00, 0.37), label='Data present')
white_patch = mpatches.Patch(color='white', label='Data absent')

ax.legend(handles=[red_patch, white_patch],loc='center left', bbox_to_anchor=(1.2, 0.7))

st.pyplot(fig)


# Clustering Analysis
st.subheader("Clustering Analysis")
# Select numeric columns for clustering
numeric_cols = ['min_weeks', 'min_weight']
clustering_df = df.dropna(subset=numeric_cols)
X = clustering_df[numeric_cols]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
clustering_df['Cluster'] = clusters

# Visualize clusters using PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_scaled)
clustering_df['PCA1'] = pca_components[:, 0]
clustering_df['PCA2'] = pca_components[:, 1]

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=clustering_df, x='PCA1', y='PCA2', hue='Cluster', style='metadata:species', ax=ax)
st.pyplot(fig)


feature_columns = ["min_weight", "max_weight", "min_weeks", "max_weeks",  "metadata:tbi_device:angle (degrees from vertical)", "metadata:tbi_device:craniectomy_size", "metadata:tbi_device:dural_tears", "metadata:tbi_device:impact_area", "metadata:tbi_device:impact_depth (mm)", "metadata:tbi_device:impact_duration (ms)", "metadata:tbi_device:impact_velocity (m/s)"]



#table
mod_columns = ["metadata:tbi_model_class", "metadata:tbi_model", "metadata:tbi_device:angle (degrees from vertical)", "metadata:tbi_device:craniectomy_size", "metadata:tbi_device:dural_tears", "metadata:tbi_device:impact_area", "metadata:tbi_device:impact_depth (mm)", "metadata:tbi_device:impact_duration (ms)", "metadata:tbi_device:impact_velocity (m/s)"]

models= df[mod_columns]
st.table(models)



# count the occurrences of each unique value 
species_counts = filtered_species["metadata:species"].value_counts()

# Display the counts as a table
st.write("Count of different species variations:")
st.write(species_counts)

# bar chart to  compare
st.write("Bar Chart of Species Variations:")
st.bar_chart(species_counts)








