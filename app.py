import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# Streamlit app title and description
st.title("CORD-19 Data Explorer")
st.write("Simple exploration of COVID-19 research papers")

# Add some basic info
st.markdown("---")
st.write("This app analyzes the CORD-19 dataset to explore COVID-19 research trends.")

# Your handle_missing function
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles NaN values in metadata_df with column-specific replacements.
    """
    replacements = {
        "title": "No Title",
        "doi": "No DOI",
        "pmcid": "No PMCID",
        "pubmed_id": "No PubMed ID",
        "abstract": "No Abstract",
        "publish_time": "1900-01-01",   # Use a default date
        "authors": "Unknown Authors",
        "journal": "Unknown Journal",
        "who_covidence_id": "No Covidence ID",
        "arxiv_id": "No ArXiv ID",
        "pdf_json_files": "No PDF JSON",
        "pmc_json_files": "No PMC JSON",
        "url": "No URL",
        "sha": "No SHA",
        "mag_id": 0,
        "s2_id": 0
    }

    # Apply
    for col, value in replacements.items():
        if col in df.columns:
            if col == "publish_time":
                df[col] = pd.to_datetime(df[col], errors="coerce").fillna(pd.to_datetime(value))
            else:
                df[col] = df[col].fillna(value)

    return df

# Load and process data
@st.cache_data
def load_and_process_data():
    """Load and process the data using your original code"""

    # Load data
    st.write("Loading data...")
    metadata_df = pd.read_csv("D:\PLP\Python_Data_Structures\week_8\Frameworks\Frameworks_Assignment\data\metadata.csv")

    # Show basic info
    st.write(f"Dataset shape: {metadata_df.shape}")

    # Handle missing data using your function
    metadata_df = handle_missing(metadata_df)

    return metadata_df

# Load the data
try:
    metadata_df = load_and_process_data()
    st.success("Data loaded successfully!")
except FileNotFoundError:
    st.error("Could not find metadata.csv file. Please make sure it's in the same directory.")
    st.stop()

# Add interactive widgets
st.sidebar.header("Settings")
year_range = st.sidebar.slider("Select year range", 2019, 2022, (2020, 2021))
top_n_journals = st.sidebar.selectbox("Top N journals to show", [5, 10, 15, 20], index=1)

# Section 1: Basic Data
st.header("📊 Dataset Overview")
col1, col2, col3 = st.columns(3)
metadata_df = load_and_process_data()
with col1:
    st.metric("Total Papers", f"{len(metadata_df):,}")

with col2:
    st.metric("Total Columns", len(metadata_df.columns))

with col3:
    unique_journals = metadata_df['journal'].nunique()
    st.metric("Unique Journals", f"{unique_journals:,}")

# Show first few rows
if st.checkbox("Show first 5 rows of data"):
    st.dataframe(metadata_df.head())

# Show missing data info
if st.checkbox("Show missing data information"):
    st.write("Missing data per column:")
    missing_data = metadata_df.isnull().sum()
    st.dataframe(missing_data[missing_data > 0])

# Section 2: Publications Over Time from  ipynb
st.header("📈 Publications Over Time")

# Count papers by publication year
publication_year_counts = metadata_df['publish_time'].dt.year.value_counts().sort_index()

# Filter by year range
filtered_counts = publication_year_counts[
    (publication_year_counts.index >= year_range[0]) &
    (publication_year_counts.index <= year_range[1])
]

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(filtered_counts.index, filtered_counts.values, marker='o')
ax.set_title("Number of COVID-19 Publications Over Time")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Publications")
ax.grid()
st.pyplot(fig)

# Show the data
st.write("Publications by year:")
year_data = pd.DataFrame({
    'Year': filtered_counts.index,
    'Count': filtered_counts.values
})
st.dataframe(year_data)

# Section 3: Top Journals
st.header("📰 Top Journals")

# Identify top journals
top_journals = metadata_df['journal'].value_counts().head(top_n_journals)

# Create bar chart
fig, ax = plt.subplots(figsize=(12, 6))
top_journals.plot(kind='bar', ax=ax)
ax.set_title(f"Top {top_n_journals} Journals Publishing COVID-19 Research")
ax.set_xlabel("Journal")
ax.set_ylabel("Number of Publications")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig)

# Show journal data
st.write("Top journals data:")
journal_data = pd.DataFrame({
    'Journal': top_journals.index,
    'Papers': top_journals.values
})
st.dataframe(journal_data)

# Section 4: Word Analysis
st.header("🔤 Most Frequent Words in Titles")

# Find most frequent
all_titles = ' '.join(metadata_df['title'].tolist()).lower()
words = re.findall(r'\b\w+\b', all_titles)
word_counts = Counter(words)
most_common_words = word_counts.most_common(20)

# Show top words
st.write("Most frequent words in titles:")
word_df = pd.DataFrame(most_common_words, columns=['Word', 'Count'])
st.dataframe(word_df)

# Word frequency bar chart
fig, ax = plt.subplots(figsize=(10, 6))
words_to_plot = word_df.head(10)  # Top 10 for better readability
ax.bar(words_to_plot['Word'], words_to_plot['Count'])
ax.set_title("Top 10 Most Frequent Words in Titles")
ax.set_xlabel("Words")
ax.set_ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# Section 5: Source Distribution
st.header("📊 Papers by Source")

if 'source_x' in metadata_df.columns:
    # Count papers by source
    source_counts = metadata_df["source_x"].value_counts()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    source_counts.plot(kind="bar", color="blue", edgecolor="black", ax=ax)
    ax.set_title("Distribution of Paper Count by Source")
    ax.set_xlabel("Source")
    ax.set_ylabel("Number of Papers")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

    # Show source data
    st.write("Papers by source:")
    source_data = pd.DataFrame({
        'Source': source_counts.index,
        'Count': source_counts.values
    })
    st.dataframe(source_data)
else:
    st.write("Source column not found in the dataset")

# Section 6: Sample Data
st.header("📋 Sample Data")
sample_size = st.slider("Number of rows to display", 5, 50, 10)

# Show sample of the actual data
st.write(f"Showing {sample_size} sample records:")
sample_columns = ['title', 'journal', 'authors', 'publish_time']
available_columns = [col for col in sample_columns if col in metadata_df.columns]

if available_columns:
    st.dataframe(metadata_df[available_columns].head(sample_size))

# Footer
st.markdown("---")
st.write("**About this app:**")
st.write("This Streamlit app uses your original analysis code to explore the CORD-19 dataset.")
st.write("Use the sidebar controls to customize the analysis.")
st.write("Data source: CORD-19 COVID-19 Research Dataset")