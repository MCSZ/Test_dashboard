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
file_path = "ModelCat_paper__020624.xlsx"
df = pd.read_excel(file_path, sheet_name='Sheet1')
df['Sex'] = pd.concat([df['Sex1'], df['Sex2']]).reset_index(drop=True)

# Streamlit Dashboard Setup
st.title("PRECISE-TBI Metadata Dashboard")

# Basic Summary
st.subheader("Metadata Summary")
categorical_columns =  ["Sex", "Species", "Strain Type", "TBI Model Type", "metadata:age:category"]



# Header
st.subheader("Summary Preview")
st.dataframe(df.head())

st.write({
    "Total Rows": len(df),
    "Total Columns": len(df.columns),
    "Missing Values": df.isnull().sum().sum(),
    "Columns with Missing Values": df.isnull().sum()[df.isnull().sum() > 0].to_dict()
})
    
#need to add more columns
for col in categorical_columns:
    if col in df.columns:
        st.subheader(f"Distribution of {col}")
        fig, ax = plt.subplots()
        sns.countplot(data=df, y="TBI Model Type", hue=col, ax=ax)
        st.pyplot(fig)

# Ensure the DataFrame contains the necessary columns
required_columns = ["Species", "Strain Type"]

if not all(col in df.columns for col in required_columns):
    st.error(required_columns)
else:
    # Drop rows with missing values in the required columns
    df = df.dropna(subset=required_columns)

    # Get unique species
    species_list = df["Species"].unique()

    # Iterate over each species and plot the distribution of strains
    for species in species_list:
        st.subheader(f"Distribution of strains for species: {species}")
        species_df = df[df["Species"] == species]
        fig, ax = plt.subplots()
        sns.countplot(y=species_df["Strain Type"], order=species_df["Strain Type"].value_counts().index, ax=ax)
        ax.set_title(f"Strain Distribution for {species}")
        st.pyplot(fig)

# General Summary
st.subheader("General Summary")
df=df.dropna(subset=['TBI Model Type'])


df_filt = df[["Sex", "Species", "Strain", "TBI Model","Age category", "Age (weeks)", "Weight (grams)", "Device Name"]]
df_filt.replace(r'^\s*$', np.nan, regex=True, inplace=True)
fig,ax = plt.subplots(figsize=(10,5))
msno.matrix(df_filt, ax=ax, fontsize=12, color= (0.93, 0.00, 0.37), sparkline=False)

red_patch = mpatches.Patch(color= (0.93, 0.00, 0.37), label='Data present')
white_patch = mpatches.Patch(color='white', label='Data absent')

ax.legend(handles=[red_patch, white_patch],loc='center left', bbox_to_anchor=(1.2, 0.7))

st.pyplot(fig)


# Age Classification Analysis
df['Age_min(weeks)'].apply(pd.to_numeric, errors='coerce')
df['Weight_min(grams)'].apply(pd.to_numeric, errors='coerce')
st.subheader("Age and Weight Distribution by Sex and Species")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x='Age_min(weeks)', y='Weight_min(grams)', hue='Sex', style='Species', ax=ax)
ax.set_xlabel('Age (weeks)')
ax.set_ylabel('Weight (grams)')
st.pyplot(fig)

# Strain Analysis
st.subheader("Strain Classification Analysis")
df['strain_prefix'] = df['Strain'].str[:4]
strain_summary = df.groupby(['strain_prefix', 'Strain']).size().reset_index(name='Count')
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=strain_summary, y='strain_prefix', hue='Strain', ax=ax)
ax.set_ylabel('Strain Prefix')
ax.set_xlabel('Count')
st.pyplot(fig)

# Model Summary
st.subheader("Model Summary by Age, Weight, Classification, Species, and Strain")
model_summary = df.groupby('metadata:tbi_model').agg({
    'Age_min(weeks)': lambda x: pd.to_numeric(x, errors='coerce').mean(),
    'Age_max(weeks)': lambda x: pd.to_numeric(x, errors='coerce').mean(),
    'Weight_min(grams)': lambda x: pd.to_numeric(x, errors='coerce').mean(),
    'Weight_min(grams)': lambda x: pd.to_numeric(x, errors='coerce').mean(),
    'metadata:sex': pd.Series.nunique,
    'metadata:species': pd.Series.nunique,
    'metadata:strain': pd.Series.nunique
}).reset_index()
model_summary.columns = ['TBI Model', 'Age_min(weeks)', 'Age_max(weeks)', 'Weight_min(grams)', 'Weight_min(grams)', 'Unique Sex Count', 'Unique Species Count', 'Unique Strain Count']
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




# count the occurrences of each unique value 
species_counts = filtered_species["metadata:species"].value_counts()

# Display the counts as a table
st.write("Count of different species variations:")
st.write(species_counts)

# bar chart to  compare
st.write("Bar Chart of Species Variations:")
st.bar_chart(species_counts)








