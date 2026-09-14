import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

# ==== CONFIGURATION ====
parser = argparse.ArgumentParser(description="Compare two experiment runs: two-way ANOVA (Dosage x Run) per trait, then per-dose t-tests and plots for significant traits.")
parser.add_argument("old_csv", help="CSV for the first (old) experiment run")
parser.add_argument("new_csv", help="CSV for the second (new) experiment run")
parser.add_argument("--output-dir", default="sig_trait_plots", help="Directory for results and plots (default: sig_trait_plots)")
parser.add_argument("--alpha", type=float, default=0.05, help="FDR significance threshold (default: 0.05)")
args = parser.parse_args()

OUTPUT_DIR = args.output_dir
ANNOT_DIR = os.path.join(OUTPUT_DIR, "annot")
ALPHA_FDR = args.alpha
# =======================

# Load datasets
old_df = pd.read_csv(args.old_csv)
new_df = pd.read_csv(args.new_csv)

# Add experiment run labels
old_df["ExperimentRun"] = "old"
new_df["ExperimentRun"] = "new"

# Combine datasets
df = pd.concat([old_df, new_df], ignore_index=True)

# Identify behavior columns (exclude metadata)
exclude_cols = {"Dosage", "Track_duration", "Track_Length", "ExperimentRun"}
behavior_columns = [
    col for col in df.columns
    if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
]

# ==== Two-way ANOVA per behavior ====
results = []
for col in behavior_columns:
    model = smf.ols(f"Q('{col}') ~ C(Dosage) * C(ExperimentRun)", data=df).fit()
    anova_table = anova_lm(model, typ=2)
    interaction_p = anova_table.loc["C(Dosage):C(ExperimentRun)", "PR(>F)"]
    results.append({"Trait": col, "Interaction_p": interaction_p})

results_df = pd.DataFrame(results)

# FDR correction
reject, pvals_corrected, _, _ = multipletests(results_df["Interaction_p"], method="fdr_bh")
results_df["FDR_p"] = pvals_corrected
results_df["Significant"] = reject

# Save ANOVA results
os.makedirs(OUTPUT_DIR, exist_ok=True)
results_df.to_csv(os.path.join(OUTPUT_DIR, "anova_interaction_results.csv"), index=False)

# Identify significant traits
sig_traits = results_df.loc[results_df["FDR_p"] <= ALPHA_FDR, "Trait"].tolist()
print(f"Significant traits (FDR ≤ {ALPHA_FDR}): {sig_traits}")

if not sig_traits:
    print("No significant interactions; nothing to plot.")
    sys.exit()

# ==== Filter common doses with ≥2 replicates per run ====
common_doses = sorted(set(df["Dosage"][df["ExperimentRun"] == "old"])
                      .intersection(df["Dosage"][df["ExperimentRun"] == "new"]))
df = df[df["Dosage"].isin(common_doses)]
cell_counts = df.groupby(["ExperimentRun", "Dosage"]).size()
bad_doses = cell_counts[cell_counts < 2].index.get_level_values("Dosage").unique()
df = df[~df["Dosage"].isin(bad_doses)]

common_doses = sorted(set(df["Dosage"][df["ExperimentRun"] == "old"])
                      .intersection(df["Dosage"][df["ExperimentRun"] == "new"]))
if not common_doses:
    raise ValueError("No overlapping doses between runs after filtering.")

# ==== Utility ====
def star(p):
    if p < 1e-4: return "****"
    if p < 1e-3: return "***"
    if p < 1e-2: return "**"
    if p < 0.05: return "*"
    return ""

# ==== Per-dose t-tests and plotting ====
os.makedirs(ANNOT_DIR, exist_ok=True)
per_dose_rows = []

for trait in sig_traits:
    raw_p = []
    dose_list = []

    # Per-dose Welch t-tests
    for d in common_doses:
        g_old = df[(df["ExperimentRun"] == "old") & (df["Dosage"] == d)][trait].dropna()
        g_new = df[(df["ExperimentRun"] == "new") & (df["Dosage"] == d)][trait].dropna()
        p = np.nan
        if len(g_old) >= 2 and len(g_new) >= 2:
            _, p = ttest_ind(g_old, g_new, equal_var=False)
        raw_p.append(p)
        dose_list.append(d)

    # FDR correction per trait
    p_arr = np.array(raw_p, dtype=float)
    q_arr = np.full_like(p_arr, np.nan, dtype=float)
    valid = ~np.isnan(p_arr)
    if valid.sum() > 0:
        _, q_valid, _, _ = multipletests(p_arr[valid], method="fdr_bh", alpha=ALPHA_FDR)
        q_arr[valid] = q_valid

    # Save results
    for d, p, q in zip(dose_list, p_arr, q_arr):
        per_dose_rows.append({"Trait": trait, "Dosage": d, "p_raw": p, "p_FDR": q})

    # Summary stats for plotting
    summ = (df.groupby(["ExperimentRun", "Dosage"], observed=True)[trait]
              .agg(mean="mean", sd="std", n="size").reset_index())
    summ["sem"] = summ["sd"] / np.sqrt(summ["n"].clip(lower=1))

    # Plot
    plt.figure(figsize=(6.5, 4.5))
    for run in ["old", "new"]:
        sub = summ[summ["ExperimentRun"] == run].sort_values("Dosage")
        if sub.empty:
            continue
        x = sub["Dosage"].astype(str)
        y = sub["mean"].values
        se = sub["sem"].values
        plt.plot(x, y, marker="o", label=run)
        plt.errorbar(x, y, yerr=se, fmt="none", ecolor="black", capsize=3)

    # Annotate stars
    look = summ.set_index(["ExperimentRun", "Dosage"])
    for d, q in zip(dose_list, q_arr):
        s = star(q)
        if not s or np.isnan(q):
            continue
        try:
            hi = max(
                look.loc[("old", d), "mean"] + look.loc[("old", d), "sem"],
                look.loc[("new", d), "mean"] + look.loc[("new", d), "sem"],
            )
        except KeyError:
            continue
        plt.text(str(d), hi * 1.05, s, ha="center", va="bottom")

    plt.title(trait)
    plt.xlabel("Dosage")
    plt.ylabel(trait)
    plt.legend(title="ExperimentRun")
    plt.tight_layout()
    plt.savefig(os.path.join(ANNOT_DIR, f"{trait}_dose_response_annotated.png"), dpi=200)
    plt.close()

# Save per-dose stats
pd.DataFrame(per_dose_rows).to_csv(os.path.join(ANNOT_DIR, "per_dose_ttests_FDR.csv"), index=False)

print(f"Annotated plots and per-dose stats saved in: {os.path.abspath(ANNOT_DIR)}")
