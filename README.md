# c-elegans_post_hoc_analysis

Data analysis scripts created during time in Sun Lab, revisiting for data science education purposes.

Statistical analysis and plotting scripts for dose-response behavioral data.

## Setup

```
pip install -r requirements.txt
```

Requires Python 3.9+ and scipy 1.11+ (for `scipy.stats.dunnett`).

## Input format

Both scripts expect a CSV with:

- a numeric `Dosage` column, where the control group is `0`
- one row per sample
- numeric columns for each behavioral trait

Non-numeric columns are ignored. Some metadata columns (e.g. `Track_duration`, `Track_Length`) are excluded from analysis; see the exclude lists at the top of each script.

## Scripts

### `dosage_heatmap.py`

For each trait, runs a one-way ANOVA across dosage groups. If it is significant (p < 0.05), runs Dunnett's test comparing each dose to control. Plots a heatmap of log2 fold change vs. control mean, annotated with significance stars.

```
python dosage_heatmap.py data.csv --title "N2 Translation @ 60"
```

### `run_comparison.py`

Compares two experiment runs of the same assay. For each trait, fits a two-way ANOVA (`Dosage * ExperimentRun`) and FDR-corrects the interaction p-values. For traits with a significant interaction, runs per-dose Welch t-tests between runs (FDR-corrected within trait) and saves annotated dose-response plots.

```
python run_comparison.py old_run.csv new_run.csv --output-dir sig_trait_plots --alpha 0.05
```

Outputs:

- `<output-dir>/anova_interaction_results.csv`
- `<output-dir>/annot/per_dose_ttests_FDR.csv`
- `<output-dir>/annot/<trait>_dose_response_annotated.png`

## Significance stars

| Stars | p |
|-------|---|
| `*` | < 0.05 |
| `**` | < 0.01 |
| `***` | < 0.001 |
| `****` | < 0.0001 |
