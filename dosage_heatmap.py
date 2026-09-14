import argparse

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from scipy.stats import dunnett
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# Columns that are not treated as behaviors
EXCLUDE_COLS = {'Dosage', 'Mean_Area','Track_duration','Track_Length','Absolute_Peristaltic_Track_Length','Absolute_Peristaltic_Speed','Peristaltic_Track_Length','Peristaltic_Speed','Max_Amplitude','Mobility_reversal_time','Mobility_idle_time','Mobility_forward_time'}

def significance_stars(p):
    if p < 0.0001:
        return '****'
    elif p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''

parser = argparse.ArgumentParser(description="Heatmap of log2 fold change vs. control per dosage, with Dunnett's test significance.")
parser.add_argument("csv", help="Path to input CSV (must contain a 'Dosage' column; control = 0)")
parser.add_argument("--title", default="", help="Plot title")
args = parser.parse_args()

# Load data into pandas df and preprocessing
df = pd.read_csv(args.csv)
behavior_columns = [col for col in df.columns if col not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(df[col])]

group_counts = df.groupby('Dosage').size()
print("Sample size (n) per dosage group:")
print(group_counts)

# Run Dunnett's test and collect p-values and creates dataframe containing the p-values in question
results = []
for col in behavior_columns:
    model = smf.ols(f"Q('{col}') ~ C(Dosage)", data=df).fit()
    anova = anova_lm(model, typ=1)
    p_anova = anova["PR(>F)"].iloc[0]
    if p_anova < 0.05:
        control = df[df["Dosage"] == 0][col]
        treatment_groups = [df[df["Dosage"] == dose][col] for dose in df["Dosage"].unique() if dose != 0]
        res = dunnett(*treatment_groups, control=control, alternative="two-sided")
        results.append(res.pvalue.tolist())
    else:
        results.append([np.nan] * (df["Dosage"].nunique() - 1))

doses = sorted(df["Dosage"].unique())
doses_wo_control = [d for d in doses if d != 0]
pval_df = pd.DataFrame(results, index=behavior_columns, columns=doses_wo_control)
pval_df.insert(0,'0',np.nan)  # Adds a control column (no significance)
annot_df = pval_df.map(significance_stars)  # Create asterisk annotation DataFrame

# Creates the heatmap matrix and applies the normalization to values
control_means = df[df['Dosage'] == 0][behavior_columns].mean()
normalized_values = df[behavior_columns] / control_means
normalized_values['Dosage'] = df['Dosage']
heatmap_matrix = np.log2(normalized_values.groupby('Dosage').mean()).round(5)
heatmap_matrix.replace(-np.inf, -4, inplace=True)

# Plot heatmap
custom_cmap = LinearSegmentedColormap.from_list("purple_white_yellow", ["purple", "white", "yellow"])
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_matrix.T, cmap=custom_cmap, center=0,
            vmin=-2, vmax=2,
            annot=annot_df,
            fmt='', linewidths=0.5, linecolor='gray')
plt.title(args.title)
plt.tight_layout()
plt.show()
