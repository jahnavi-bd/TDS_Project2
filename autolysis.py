import os
import pandas as pd
import chardet
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from scipy.stats import zscore
from PIL import Image
import openai
import argparse

# Set up OpenAI API key from environment variable
import os
import openai

# Retrieve the API key from the environment variable
api_key = os.getenv("AI_PROXY")
if not api_key:
    raise EnvironmentError("AI_PROXY is not set in the environment.")

openai.api_key = api_key

# Load CSV file using chardet to detect the encoding
def load_data(file_name):
    try:
        with open(file_name, 'rb') as file:
            result = chardet.detect(file.read())
        encoding = result['encoding']
        df = pd.read_csv(file_name, encoding=encoding)
        return df
    except Exception as e:
        print(f"Error loading file {file_name}: {e}")
        return None

# Perform generic analysis
def generic_analysis(df):
    summary_stats = df.describe()
    missing_values = df.isnull().sum()[df.isnull().sum() > 0].index.tolist()
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numeric_df.corr() if not numeric_df.empty else None
    return summary_stats, missing_values, correlation_matrix

# Detect outliers using Z-score
def detect_outliers(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    z_scores = zscore(df[numeric_cols].dropna())
    outliers = (abs(z_scores) > 3).sum(axis=0)
    outlier_columns = numeric_cols[(abs(z_scores) > 3).any(axis=0)].tolist()
    return outliers, outlier_columns

# Perform clustering
def perform_clustering(df, n_clusters=3):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df_cleaned = df[numeric_cols].dropna()
    df_scaled = StandardScaler().fit_transform(df_cleaned)
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(df_scaled)
    df['Cluster'] = None
    df.loc[df_cleaned.index, 'Cluster'] = clusters
    return df

# Detect anomalies
def detect_anomalies(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df_cleaned = df[numeric_cols].dropna()
    df_scaled = StandardScaler().fit_transform(df_cleaned)
    model = IsolationForest(contamination=0.1)
    anomalies = model.fit_predict(df_scaled)
    df['Anomaly'] = None
    df.loc[df_cleaned.index, 'Anomaly'] = anomalies
    return df

# Generate a correlation heatmap
def create_heatmap(correlation_matrix, output_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.savefig(output_path)
    plt.close()

# Generate a boxplot
def create_boxplot(df, output_path):
    plt.figure(figsize=(10, 8))
    sns.boxplot(data=df.select_dtypes(include=['float64', 'int64']).dropna())
    plt.title("Outliers Boxplot")
    plt.savefig(output_path)
    plt.close()

# Resize image
def resize_image(image_path, size=(512, 512)):
    with Image.open(image_path) as img:
        img = img.resize(size)
        img.save(image_path)

# Generate analysis summary using LLM
def generate_story(dataset_name, df, missing_values, outliers_detected, correlations_found):
    llm_prompt = f"""
    Dataset: {dataset_name}
    Columns: {list(df.columns)}
    Missing values in: {missing_values}
    Outliers detected in: {outliers_detected}
    Correlations found: {list(correlations_found.columns) if correlations_found is not None else 'None'}.
    Provide insights and recommendations.
    """
    try:
        response = openai.Completion.create(
            engine="gpt-4o-mini",
            prompt=llm_prompt,
            max_tokens=100,
            temperature=0.7
        )
        return response['choices'][0]['text'].strip()
    except Exception as e:
        print(f"Error generating story from LLM: {e}")
        return "Analysis summary could not be generated."

# Create README.md
def create_readme(df, analysis_summary, missing_values, outliers_detected, output_path):
    readme_content = f"""
# Data Analysis Report

## Overview:
Columns: {list(df.columns)}

## Analysis Summary:
{analysis_summary}

### Missing Values:
{missing_values}

### Outliers Detected:
{outliers_detected}

## Visualizations:
![Correlation Heatmap](correlation_heatmap.png)
![Outliers Boxplot](outlier_boxplot.png)
"""
    with open(output_path, "w") as file:
        file.write(readme_content)

# Main function
def main():
    parser = argparse.ArgumentParser(description="Run analysis on a CSV file.")
    parser.add_argument("file_name", help="Path to the CSV file")
    args = parser.parse_args()

    df = load_data(args.file_name)
    if df is None:
        return

    dataset_name = os.path.basename(args.file_name).split('.')[0]
    summary_stats, missing_values, correlation_matrix = generic_analysis(df)
    outliers_detected, outlier_columns = detect_outliers(df)
    df_with_clusters = perform_clustering(df)
    df_with_anomalies = detect_anomalies(df)

    heatmap_path = f"{dataset_name}_correlation_heatmap.png"
    boxplot_path = f"{dataset_name}_outlier_boxplot.png"
    readme_path = "README.md"

    if correlation_matrix is not None:
        create_heatmap(correlation_matrix, heatmap_path)
    create_boxplot(df, boxplot_path)
    resize_image(heatmap_path)
    resize_image(boxplot_path)

    analysis_summary = generate_story(dataset_name, df, missing_values, outlier_columns, correlation_matrix)
    create_readme(df, analysis_summary, missing_values, outlier_columns, readme_path)

    print("Analysis complete. Files saved.")

if __name__ == "__main__":
    main()
