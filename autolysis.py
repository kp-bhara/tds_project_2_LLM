# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "openai",
#   "markdown",
#   "seaborn",
#   "numpy"
# ]
# ///



import json
import os
import sys
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from openai import OpenAI


def load_data(file_path):
    """Load CSV file into a Pandas DataFrame."""
    try:
        df = pd.read_csv(file_path, encoding = 'unicode_escape')
        print("Data loaded successfully!")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)



def create_output_folder(file_path):
    """Create an output folder named after the dataset."""
    dataset_name = os.path.splitext(os.path.basename(file_path))[0]
    output_folder = dataset_name
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return output_folder



def clean_urls_from_categorical(df):
    """Clean URL values from categorical columns."""
    url_pattern = r'(https?://\S+|www\.\S+)'  # Regex for URLs
    categorical_columns = df.select_dtypes(include=['object']).columns

    for col in categorical_columns:
        df[col] = df[col].apply(lambda x: np.nan if re.match(url_pattern, str(x)) else x)

    print("URL-like values have been replaced with NaN in categorical columns.")
    return df

def analyze_data(df):
    """Perform concise data analysis on the DataFrame."""
    # Clean URLs in categorical data
    df = clean_urls_from_categorical(df)

    df.columns = df.columns.str.replace(' ', '_')


    # Numeric analysis
    numeric_columns = df.select_dtypes(include=['number']).columns
    numeric_summary = df[numeric_columns].describe().to_string()

    # Categorical analysis (limit value counts to top 5)
    categorical_columns = df.select_dtypes(include=['object']).columns
    categorical_summary = df[categorical_columns].describe().to_string()
    categorical_value_counts = {
        col: df[col].value_counts().head(5).to_string() for col in categorical_columns
    }

    # Missing values summary
    missing_values = df.isnull().sum().to_string()

    # Correlation analysis: Calculate correlation matrix and extract the top 5 correlations
    correlation_matrix = df[numeric_columns].corr()
    # Unstack the correlation matrix, sort values, and remove the diagonal (correlation of a feature with itself)
    correlation_pairs = correlation_matrix.unstack().sort_values(ascending=False)
    correlation_pairs = correlation_pairs[correlation_pairs < 1]  # Remove the diagonal (self-correlation)
    top_5_correlations = correlation_pairs.head(5)

    # Generating the final analysis text
    analysis = "### Summary Statistics (Numerical Data):\n"
    analysis += numeric_summary + "\n\n"

    analysis += "### Summary Statistics (Categorical Data):\n"
    analysis += categorical_summary + "\n\n"

    analysis += "### Top 5 Value Counts for Categorical Columns:\n"
    for col, counts in categorical_value_counts.items():
        analysis += f"\n{col}:\n{counts}\n"

    analysis += "\n### Missing Values:\n" + missing_values + "\n\n"

    analysis += "### Top 5 Highest Correlations:\n"
    for (col1, col2), corr_value in top_5_correlations.items():
        analysis += f"{col1} & {col2}: {corr_value:.4f}\n"

    return analysis, df




def get_relevant_columns_for_visualization(context):
    """Ask LLM to determine the most relevant columns for visualization."""
    client = OpenAI(
        base_url="https://aiproxy.sanand.workers.dev/openai/v1",
        api_key=os.environ.get("AIPROXY_TOKEN")
    )

    retries = 3  # Retry up to 3 times
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a data / business analyst who specializes in determining the most relevant numeric column and categorical column for data visualization that holds significant business meaning and makes the most sense for analysis purposes"},
                    {"role": "user", "content": f"Here is a summary of the dataset:\n{context}\nIdentify the most relevant numeric column and categorical column suitable for visualization that can highlight something meaningful and that holds significance for a business/perception analysis. Reply ONLY in JSON format with two keys: 'numeric_column' (list of only relevant numeric column name). 'categorical_column' (list of only relevant categorical column name). You can use cues from the dataset description for some hint. Do not include any additional text or explanations. You must return only one column name for each, avoiding irrelevant or low-impact numeric and categorical columns. Do not include other numerical and categorical columns that are not useful and relevant for a business or sentiment analysis"}
                ]   
            )
            
            # Extract raw response content
            raw_content = response.choices[0].message.content
            
            # Clean the response content (strip backticks or other non-JSON text)
            cleaned_content = raw_content.strip("`").strip("json").strip()
            
            # Parse cleaned JSON
            relevant_columns = json.loads(cleaned_content)
            print("&^#$^@&#$^#$^@$&#*^#$&^#%@&#%$!^#@$%&$^*@%&#^!%#@&%$@^#%@#")
            print(relevant_columns)
            return relevant_columns
        

            # try:
            #     relevant_columns = json.loads(response.choices[0].message.content)
            # except json.JSONDecodeError as e:
            #     print(f"Error parsing LLM response as JSON: {e}")
            #     print("Response received:", response.choices[0].message.content)
            #     sys.exit(1)

            #relevant_columns = json.loads(response.choices[0].message.content) # Parse the LLM response return
            #relevant_columns = response.choices[0].message.content
            #return relevant_columns
        except Exception as e:
            if "429" in str(e):
                print(f"Rate limit exceeded. Retrying in 15 seconds... (Attempt {attempt+1}/{retries})")
                time.sleep(15)
            else:
                print(f"Error in LLM column selection: {e}")
                return {"numeric_columns": []}
    return {"numeric_columns": []}  # After retries, return an empty dictionary if still failing.





def generate_visualizations(df, relevant_columns, output_folder):
    """Generate and save charts based on the relevant columns."""
    visualizations = []

    # Correlation Heatmap
    # Correlation Heatmap
    numeric_columns = df.select_dtypes(include=['number']).columns  # **Generate correlation matrix for all numeric columns**
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[numeric_columns].corr()
    
    # Use pandas categorical check to avoid deprecation warning
    mask = np.zeros_like(correlation_matrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, 
                mask=mask, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title("Correlation Matrix")
    plt.tight_layout()
    filename = os.path.join(output_folder, "correlation_matrix.png")
    plt.savefig(filename)
    plt.close()
    visualizations.append(filename)


# Bar Plots for Categorical Columns
    for col in relevant_columns['categorical_column']:
        top_values = df[col].value_counts().head(5)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_values.index, y=top_values.values, palette="viridis")
        plt.title(f"Top 5 Values in '{col}'")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        #filename = f"top_values_{col}.png"
        filename = os.path.join(output_folder, f"top_values_{col}.png")
        plt.savefig(filename)
        plt.close()
        visualizations.append(filename)

    # Histogram for Relevant Numeric Columns
    for col in relevant_columns['numeric_column']:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col].dropna(), bins=30, kde=True, color="skyblue")
        plt.title(f"Distribution of '{col}'")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        filename = os.path.join(output_folder, f"histogram_{col}.png")
        plt.savefig(filename)
        plt.close()
        visualizations.append(filename)

    return visualizations





def ask_llm_for_analysis(context):
    """Send context to GPT-4o-Mini for deeper analysis or insights."""
    client = OpenAI(
        base_url="https://aiproxy.sanand.workers.dev/openai/v1",
        api_key=os.environ.get("AIPROXY_TOKEN")
    )

    retries = 3  # Retry up to 3 times
    for attempt in range(retries):


        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a data analyst who specializes in creating comprehensive reports based on information from a dataset."},
                    {"role": "user", "content": f"Here's the data summary:\n{context}\nProvide insightful analysis and key takeaways in detail."}
                ]
            )
            return response.choices[0].message.content
        

        except Exception as e:
            if "429" in str(e):
                print(f"Rate limit exceeded. Retrying in 15 seconds... (Attempt {attempt+1}/{retries})")
                time.sleep(15)

            else:
                print(f"Error in LLM analysis: {e}")
                return "Unable to generate LLM insights."
    return "Exceeded retry limit. Unable to complete request."

            





def generate_story(analysis_results, visualizations, output_folder):
    """Generate a Markdown narrative using LLM."""
    client = OpenAI(
        base_url="https://aiproxy.sanand.workers.dev/openai/v1",
        api_key=os.environ.get("AIPROXY_TOKEN")
    )

    retries = 3  # Retry up to 3 times
    for attempt in range(retries):

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a data storyteller who creates engaging narratives from data analysis."},
                    {"role": "user", "content": f"Here is the analysis:\n{analysis_results}.\nCreate a compelling report that explains the data, its insights, and potential implications. Include some statistical rigor to the report.Explain the insights and implications to great depth, but in an understandable way. Note that the first 2 headings must be an introduction, followed with the dataset description ; wherein you talk about the column name, type and what it means ; and then move on to the rest. Keep the tone slightly formal but compelling and understandable."}
                ]
            )
            story = response.choices[0].message.content

            readme_path = os.path.join(output_folder, "README.md")
            # Write README.md
            with open(readme_path, "w", encoding ="utf-8") as f:
                f.write("# Automated Data Analysis Report\n\n")
                f.write(story)
                f.write("\n\n## Visualizations\n")
                for img in visualizations:
                    f.write(f"![{img}]({img})\n")
                return story

        except Exception as e:
            if "429" in str(e):
                print(f"Rate limit exceeded. Retrying in 15 seconds... (Attempt {attempt+1}/{retries})")
                time.sleep(15)
        
            else:
                print(f"Error generating story: {e}")
                readme_path = os.path.join(output_folder, "README.md")
                with open(readme_path, "w") as f:
                    f.write("# Automated Data Analysis Report\n\n")
                    f.write("Unable to generate AI-powered narrative.\n")
                return "Unable to generate story."
    return "Exceeded retry limit. Unable to complete request."        





def main():
    # Check if a CSV file is provided as an argument
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    print(f"Loading data from {file_path}...")

    output_folder = create_output_folder(file_path)


    # Step 1: Load data
    df = load_data(file_path)

    # Step 2: Analyze data
    print("Analyzing data...")
    analysis_results, analyzed_df = analyze_data(df)
    print(analysis_results)

    # Identify numeric and categorical columns
    numeric_columns = analyzed_df.select_dtypes(include=['number']).columns
    categorical_columns = analyzed_df.select_dtypes(include=['object']).columns

    # Step 3: Determine relevant columns for visualization
    print("Determining relevant columns for visualization...")
    relevant_columns = get_relevant_columns_for_visualization(analysis_results)

    # Step 4: Visualize data
    print("Generating visualizations...")
    visualizations = generate_visualizations(analyzed_df, relevant_columns, output_folder)

    # Step 5: Ask LLM for deeper insights
    print("Requesting insights from GPT-4o-mini...")
    insights = ask_llm_for_analysis(analysis_results)
    print("\nAI Insights:\n", insights)

    # Step 6: Generate story/report
    print("Generating story report...")
    generate_story(analysis_results, visualizations, output_folder)
    print("Report saved as README.md")

if __name__ == "__main__":
    main()
