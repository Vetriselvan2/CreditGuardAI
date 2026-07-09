# CreditGuard AI — Part 1: Data Cleaning & Exploratory Data Analysis

## Overview

This project analyzes the **Credit Card Approval Prediction** dataset (`application_record.csv`) as the first stage of a four-part capstone pipeline that goes from raw data to a deployed AI system. This Part focuses on **data cleaning, feature engineering, statistical analysis, and visualization**, producing a cleaned dataset that feeds into Part 2 (supervised ML).

The dataset contains customer-level records (income, education, employment, family status, age, etc.) used to study patterns relevant to credit risk analysis. It has well over 500 rows and includes both numeric columns (e.g. income, age) and categorical columns (e.g. education type), satisfying the structured-dataset requirement.

## What the Code Does

The notebook (`part_1.ipynb`) performs the following steps in order:

1. **Library setup** — imports pandas, numpy, matplotlib, seaborn; creates a `plots/` folder for saved figures.
2. **Data loading** — reads the raw CSV into a DataFrame and previews it.
3. **Initial inspection** — checks shape and column data types.
4. **Missing value analysis** — computes missing count and percentage per column; flags columns with >20% missing data.
5. **Missing value treatment** — numeric columns with <20% missing values are median-imputed (median is used because financial data is prone to extreme values); columns with >20% missing values are dropped entirely.
6. **Duplicate handling** — detects and removes fully duplicated rows.
7. **Data type optimization** — converts `NAME_EDUCATION_TYPE` from `object` to `category` dtype to reduce memory usage, with a before/after memory comparison.
8. **Feature engineering** — converts the raw `DAYS_BIRTH` field (stored as negative days) into a human-readable `AGE` column.
9. **Descriptive statistics** — generates summary statistics (`df.describe()`).
10. **Skewness analysis** — computes skewness for all numeric columns and identifies the most skewed feature.
11. **Outlier detection (IQR method)** — calculates Q1, Q3, IQR, and outlier bounds for `AMT_INCOME_TOTAL` and `AGE`. Outliers are flagged but **not removed**, since extreme income values can represent genuine customers.
12. **Visualizations** (5 required plots, all saved to `plots/`):
    - Line plot — customer age trend across records
    - Bar chart — average income by education level
    - Histogram — distribution of the most-skewed numeric feature
    - Scatter plot — age vs. income relationship
    - Box plot — income distribution across education categories
13. **Correlation analysis**:
    - Pearson correlation heatmap across numeric features, with the highest-correlated pair identified
    - Spearman rank correlation, compared against Pearson to surface non-linear/monotonic relationships (top 3 differences reported)
14. **Mean vs. median comparison** — for the two most-skewed columns, to justify the imputation strategy used.
15. **Grouped aggregation** — mean, standard deviation, and record count of income grouped by education level, highlighting which group has the highest mean income and the most internal variation.
16. **Export** — saves the fully cleaned dataset as `cleaned_credit_data.csv` for use in Part 2.

## Repository Structure

```
part1/
├── README.md
├── part_1.ipynb              # Main analysis notebook (outputs cleared)
├── application_record.csv    # Raw input dataset
├── cleaned_credit_data.csv   # Cleaned output dataset (generated on run)
└── plots/
    ├── age_line_plot.png
    ├── education_income_bar.png
    ├── skew_histogram.png
    ├── age_income_scatter.png
    ├── income_boxplot.png
    └── correlation_heatmap.png
```

## Dependencies

- Python 3.9+
- pandas
- numpy
- matplotlib
- seaborn

Install with:

```bash
pip install pandas numpy matplotlib seaborn
```

No API keys or secrets are required for this Part, so no `.env` file is needed.

## How to Run

1. Clone the repository and navigate to the `part1/` folder.
2. Place `application_record.csv` in the same folder as the notebook (update the file path in the "Load Dataset" cell if your dataset is stored elsewhere — the notebook currently reads from a local path).
3. Launch Jupyter and run all cells top to bottom:
   ```bash
   jupyter notebook part_1.ipynb
   ```
   or, to run non-interactively:
   ```bash
   jupyter nbconvert --to notebook --execute part_1.ipynb
   ```
4. On completion, the notebook will:
   - Print inspection, missing-value, skewness, outlier, and correlation summaries to output cells
   - Save six `.png` plots into `plots/`
   - Save the cleaned dataset as `cleaned_credit_data.csv`

## Key Findings & Design Decisions

- **Imputation strategy**: Median imputation was chosen over mean for numeric columns because financial variables like income are right-skewed and contain extreme values that would distort a mean-based fill.
- **Column dropping threshold**: Columns missing more than 20% of their values were dropped rather than imputed, since heavy imputation at that scale would introduce unreliable, low-signal data.
- **Outliers retained**: Outliers in income and age were detected via the IQR method but deliberately kept in the dataset, as they may represent legitimate high-income or older customers rather than data errors.
- **Correlation approach**: Both Pearson (linear) and Spearman (monotonic/rank-based) correlations were computed. Their differences highlight relationships that are monotonic but not strictly linear — relevant for financial data, which often behaves non-linearly.
- **Correlation ≠ causation**: The most strongly correlated variable pair was identified, but the analysis explicitly notes that a third factor (e.g. employment status or education) could be driving both variables rather than a direct causal link.
- **Group analysis**: Comparing income across education levels showed that while some education groups have a higher mean income, high standard deviation within groups indicates education alone is not a strong sole predictor of income — supporting the need for a multi-feature model in Part 2.

## Notes

- The notebook's outputs are cleared before submission, per assignment rules; re-running all cells reproduces all printed statistics and regenerates all plots.
- This dataset and cleaning pipeline form the foundation for Part 2, where the cleaned CSV will be used to train supervised machine learning models.
