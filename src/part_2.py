import pandas as pd
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

def load_data(filename):
    df = pd.read_csv(filename)
    df['label'] = df['label'].str.extract('(\d+)').astype(int)

    print("Data loaded!")
    return df

def calculate_agreement(df, agreement_filename):
    print("Calculating agreement...")
    
    df['coder_agree'] = (df['ertekeles_N'] == df['ertekeles_A']).astype(int)
    
    df['model_agree'] = 0
    condition_both_agree = (df['label'] == df['ertekeles_N']) & (df['label'] == df['ertekeles_A'])
    condition_one_agree = (df['label'] == df['ertekeles_N']) | (df['label'] == df['ertekeles_A'])
    df.loc[condition_both_agree, 'model_agree'] = 2
    df.loc[condition_one_agree & ~condition_both_agree, 'model_agree'] = 1

    coder_agree_counts = df['coder_agree'].value_counts()
    percentage_agreement_coder = coder_agree_counts[1] / coder_agree_counts.sum() * 100

    model_agree_counts = df['model_agree'].value_counts()
    percentage_agreement_model_both = model_agree_counts[2] / model_agree_counts.sum() * 100
    percentage_agreement_model_one = (model_agree_counts[1] + model_agree_counts[2]) / model_agree_counts.sum() * 100

    with open(agreement_filename, 'w') as file:
        print(f"Coder agreement with eachother: {percentage_agreement_coder}%", file=file)
        print(f"Model agreement with both annotators: {percentage_agreement_model_both}%", file=file)
        print(f"Model agreement with only one annotator: {percentage_agreement_model_one}%", file=file)

def prepare_input(df):
    print("Reorganizing data frame...")

    def is_neutral(value):
        if value == 1:
            return 1
        else:
            return 0

    df['l_neu'] = df['label'].apply(is_neutral)
    df['n_neu'] = df['ertekeles_N'].apply(is_neutral)
    df['a_neu'] = df['ertekeles_A'].apply(is_neutral)

def analyze(df, chisq_filename, plot_filename):
    print("Analyzing sentiment...")

    contingency_table = pd.crosstab(
        df['l_neu'], 
        [df['n_neu'], 
        df['a_neu']])
    
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    expected_df = pd.DataFrame(
        expected, 
        index=df['l_neu'].unique(), 
        columns=pd.MultiIndex.from_product([
            df['n_neu'].unique(), 
            df['a_neu'].unique()]))
    
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
    axes[0].set_xlabel('Annotator ratings')
    axes[0].set_ylabel('Model ratings')

    sns.heatmap(
        expected_df, 
        annot=True, 
        cmap='viridis', 
        fmt='.2f', 
        cbar_kws={'label': 'Expected Frequencies'}, 
        ax=axes[1])
    axes[1].set_title('Expected Frequencies')
    axes[1].set_xlabel('Annotator ratings')
    axes[1].set_ylabel('Model ratings') 

    plt.savefig(plot_filename)
    print("Analysis complete! Please check the /out folder.")

def main():
    file_name = "data/coder_data.csv"
    agreement_file_name = "out/agreement"
    chisq_file_name = "out/analysis_2"
    plot_file_name = "out/plot_2"

    data = load_data(file_name)
    calculate_agreement(data, agreement_file_name)
    prepare_input(data)
    analyze(data, chisq_file_name, plot_file_name)

if __name__ == "__main__":
    main()