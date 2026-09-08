import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

def detect_target_col(df):
    candidates = ['diagnosis', 'target', 'class', 'label']
    for col in candidates:
        if col in df.columns:
            return col
    # fallback to last non-id column
    non_id = [c for c in df.columns if c.lower() != 'id' and not c.startswith('Unnamed')]
    return non_id[-1] if non_id else None

def perform_eda(filepath, output_md):
    df = pd.read_csv(filepath)
    
    # Pre-clean
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    empty_cols = [c for c in df.columns if df[c].isnull().all() or "Unnamed" in c]
    if empty_cols:
        df = df.drop(empty_cols, axis=1)
        
    target_col = detect_target_col(df)
    
    # Encode target if it's categorical or non-numeric
    if target_col and not pd.api.types.is_numeric_dtype(df[target_col]):
        df['target_encoded'] = df[target_col].astype('category').cat.codes
    elif target_col:
        df['target_encoded'] = df[target_col]
        
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    numerical_cols = [c for c in numerical_cols if c != 'target_encoded']

    with open(output_md, 'w') as f:
        f.write("# Advanced Exploratory Data Analysis Report\n\n")
        
        # 1. Basic Info
        f.write("## 1. Basic Information\n")
        f.write(f"- **Total Samples:** {df.shape[0]}\n")
        f.write(f"- **Total Features:** {df.shape[1]}\n")
        f.write(f"- **Detected Target Column:** `{target_col}`\n\n")
        
        # 2. Missing Values
        f.write("## 2. Missing Values\n")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) == 0:
            f.write("No missing values found in the dataset.\n\n")
        else:
            f.write("```text\n")
            f.write(str(missing) + "\n")
            f.write("```\n\n")
            
        # 3. Near-Zero Variance Analysis
        f.write("## 3. Near-Zero Variance Analysis\n")
        f.write("Features with zero or near-zero variance carry little to no information.\n")
        variances = df[numerical_cols].var()
        near_zero = variances[variances < 1e-4]
        if not near_zero.empty:
            f.write("```text\n")
            f.write("Features with variance < 1e-4:\n")
            f.write(str(near_zero) + "\n")
            f.write("```\n\n")
        else:
            f.write("No near-zero variance features found.\n\n")
            
        # 4. IQR-Based Outlier Analysis
        f.write("## 4. IQR-Based Outlier Analysis\n")
        f.write("Outliers detected using the Interquartile Range (IQR) method (1.5 * IQR).\n")
        f.write("```text\n")
        Q1 = df[numerical_cols].quantile(0.25)
        Q3 = df[numerical_cols].quantile(0.75)
        IQR = Q3 - Q1
        outliers_count = ((df[numerical_cols] < (Q1 - 1.5 * IQR)) | (df[numerical_cols] > (Q3 + 1.5 * IQR))).sum()
        outliers_percent = (outliers_count / df.shape[0]) * 100
        outliers_df = pd.DataFrame({'Outlier Count': outliers_count, 'Percentage (%)': outliers_percent})
        outliers_df = outliers_df[outliers_df['Outlier Count'] > 0].sort_values(by='Percentage (%)', ascending=False)
        f.write(outliers_df.to_string() + "\n")
        f.write("```\n\n")
        
        # 5. Target Class Distribution
        f.write("## 5. Target Class Distribution\n")
        if target_col:
            f.write("```text\n")
            f.write(str(df[target_col].value_counts()) + "\n\n")
            f.write(str(df[target_col].value_counts(normalize=True) * 100) + " %\n")
            f.write("```\n\n")
            
            # Plot Target Distribution
            plt.figure(figsize=(6, 4))
            sns.countplot(x=target_col, data=df, palette='viridis')
            plt.title('Target Class Distribution')
            plt.savefig('target_distribution.png', bbox_inches='tight')
            plt.close()
            f.write("![Target Distribution](target_distribution.png)\n\n")
            
        # 6. Target Correlation Analysis
        f.write("## 6. Target Correlation Analysis\n")
        if 'target_encoded' in df.columns:
            f.write("Correlation of numeric features with the encoded target variable:\n")
            target_corr = df[numerical_cols].apply(lambda x: x.corr(df['target_encoded']))
            target_corr = target_corr.sort_values(key=abs, ascending=False)
            f.write("```text\n")
            f.write(str(target_corr) + "\n")
            f.write("```\n\n")
            
            # Correlation Heatmap
            plt.figure(figsize=(12, 10))
            corr_matrix = df[numerical_cols].corr()
            sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False)
            plt.title('Feature Correlation Heatmap')
            plt.savefig('correlation_heatmap.png', bbox_inches='tight')
            plt.close()
            f.write("![Correlation Heatmap](correlation_heatmap.png)\n\n")
            
        # 7. Class-wise Feature Distributions
        f.write("## 7. Class-wise Feature Distributions\n")
        f.write("Visualizing how the distributions of the top 12 most correlated features differ between classes.\n\n")
        if 'target_encoded' in df.columns and len(numerical_cols) >= 12:
            top_features = target_corr.head(12).index.tolist()
            
            fig, axes = plt.subplots(4, 3, figsize=(15, 15))
            axes = axes.flatten()
            for i, feature in enumerate(top_features):
                sns.kdeplot(data=df, x=feature, hue=target_col, fill=True, ax=axes[i], common_norm=False, palette='viridis')
                axes[i].set_title(f'Distribution of {feature}')
            
            plt.tight_layout()
            plt.savefig('classwise_distributions.png', bbox_inches='tight')
            plt.close()
            f.write("![Class-wise Feature Distributions](classwise_distributions.png)\n\n")
            
if __name__ == "__main__":
    filepath = 'c:/Users/Himanshi/OneDrive/Dokumen/numerical breast cancer/breast-cancer-wisconsin-data.csv'
    output = 'eda_report.md'
    if os.path.exists(filepath):
        print("Running Advanced EDA...")
        perform_eda(filepath, output)
        print(f"EDA Summary saved to {output} along with PNG visualizations.")
    else:
        print(f"File {filepath} not found!")
