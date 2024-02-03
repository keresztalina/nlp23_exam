import pandas as pd
import nltk
from transformers import pipeline
import torch
import string
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

def load_data(filename):
    df = pd.read_csv(filename)
    df = df.rename(columns={'full text': 'text'})

    print("Data loaded!")
    return df

def create_input_df(df):
    print("Filtering data and preparing model input...")

    def prepare_data(df, keyword: str):

        def filter_text(df, keyword):
            result = df[df['text'].str.contains(
                keyword, 
                case=True, 
                na=False)]

            return result

        filtered_df = filter_text(df, keyword)
        filtered_df['input_text'] = keyword + " [SEP] " + filtered_df['text']
        prepared_df = filtered_df[filtered_df['input_text'].str.len() <= 256]
        
        return prepared_df

    brusszel = prepare_data(df, "Brüsszel")
    eu = prepare_data(df, "EU")
    europai_unio = prepare_data(df, "Európai Unió")
    az_unio = prepare_data(df, "az unió")

    input_df = pd.concat([brusszel, eu, europai_unio, az_unio])

    print("Data filtered and model input prepared!")
    return input_df

def classify(df):
    print("Loading classifier...")

    classifier = pipeline(
        task="sentiment-analysis", 
        model="NYTK/sentiment-ohb3-hubert-hungarian", 
        framework="pt") # coercing it to use pytorch

    print("Beginning classification...")
    df['output'] = df['input_text'].apply(lambda text: classifier(text)[0])
    df[['label', 'score']] = df['output'].apply(lambda x: pd.Series([x['label'], x['score']]))
    df = df.drop('output', axis=1)

    print("Classification complete!")
    return df

def analyze(df, chisq_filename, plot_filename):
    print("Analyzing sentiment...")

    df['label'] = df['label'].str.extract('(\d+)').astype(int)
    
    contingency_table = pd.crosstab(df['source'], df['label'])
    
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    expected_df = pd.DataFrame(
        expected, 
        index=df['source'].unique(), 
        columns=df['label'].unique())
    
    with open(chisq_filename, 'w') as file:
        print(f"Chi-Square Value: {chi2}", file=file)
        print(f"P-value: {p}", file=file)
        print(f"Degrees of Freedom: {dof}", file=file)
        print("Observed Frequencies:", file=file)
        print(contingency_table, file=file)
        print("Expected Frequencies:", file=file)
        print(expected_df, file=file)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    sns.heatmap(
        contingency_table, 
        annot=True, 
        cmap='viridis', 
        fmt='.2f', 
        cbar_kws={'label': 'Observed Frequencies'}, 
        ax=axes[0])
    axes[0].set_title('Observed Frequencies')
    axes[0].set_xlabel('Sentiment rating')
    axes[0].set_ylabel('Source')
    axes[0].set_xticklabels(['negative', 'neutral', 'positive'])
    axes[0].set_yticklabels(['Origo.hu', 'Telex.hu'])

    sns.heatmap(
        expected_df, 
        annot=True, 
        cmap='viridis', 
        fmt='.2f', 
        cbar_kws={'label': 'Expected Frequencies'}, 
        ax=axes[1])
    axes[1].set_title('Expected Frequencies')
    axes[1].set_xlabel('Sentiment rating')
    axes[1].set_ylabel('Source')
    axes[1].set_xticklabels(['negative', 'neutral', 'positive'])
    axes[1].set_yticklabels(['Origo.hu', 'Telex.hu'])

    plt.savefig(plot_filename)
    print("Analysis complete! Please check the /out folder.")

def main():

    file_name = "data/data_exp.csv"
    chisq_file_name = "out/analysis_1"
    plot_file_name = "out/plot_1"
    
    data = load_data(file_name)
    data2 = create_input_df(data) 
    data3 = classify(data2)
    analyze(data3, chisq_file_name, plot_file_name)

if __name__ == "__main__":
    main()
