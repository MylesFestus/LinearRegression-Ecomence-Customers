# E-Commerce Customers — Machine Learning Analysis

Predict how much a customer will spend annually based on their engagement behaviour using five regression models.

---

## Dataset

| Property | Detail |
|---|---|
| File | `Ecommerce_Customers` (CSV) |
| Rows | 500 customers |
| Raw columns | 8 (Email, Address, Avatar dropped — non-numeric) |
| Working features | 4 numeric predictors |
| Target | `Yearly Amount Spent` ($) |
| Missing values | None |

### Features

| Feature | Description | Correlation with Target |
|---|---|---|
| `Length of Membership` | Years the customer has been a member | **+0.809** (strongest) |
| `Time on App` | Avg. minutes spent on the mobile app | +0.499 |
| `Avg. Session Length` | Avg. minutes per session | +0.355 |
| `Time on Website` | Avg. minutes spent on the website | −0.003 (negligible) |

---

## Project Structure

```
.
├── Ecommerce_Customers          # Raw dataset (CSV)
├── ecommerce_ml.py              # Main analysis script
├── README.md                    # This file
├── 01_distributions.png         # Feature + target histograms
├── 02_pairplot.png              # All-variable pairplot
├── 03_correlation_heatmap.png   # Pearson correlation heatmap
├── 04_scatter_plots.png         # Feature vs target scatter plots
├── 05_model_comparison.png      # R² and RMSE bar charts
├── 06_cross_validation.png      # 5-fold CV R² with error bars
├── 07_actual_vs_predicted.png   # Actual vs predicted — all models
├── 08_lr_coefficients.png       # Linear regression coefficients
├── 09_feature_importance.png    # RF and GB feature importances
└── 10_residual_analysis.png     # Residual plots for best model
```

---

## Setup & Installation

### Requirements

```
Python >= 3.8
```

### Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

### Run the analysis

Place `Ecommerce_Customers` in the same directory as the script, then:

```bash
python ecommerce_ml.py
```

All 10 output plots are saved automatically to the working directory.

---

## Models

Five regression algorithms are trained and compared:

| Model | Type | Notes |
|---|---|---|
| **Linear Regression** | Linear | Baseline; highly interpretable |
| **Ridge** | Linear + L2 regularisation | Reduces overfitting via weight penalty |
| **Lasso** | Linear + L1 regularisation | Can zero out irrelevant features |
| **Random Forest** | Ensemble (bagging) | 100 trees; captures non-linearity |
| **Gradient Boosting** | Ensemble (boosting) | 100 trees; sequential error correction |

---

## Results

### Test Set Performance

| Model | R² | RMSE ($) | MAE ($) | CV R² |
|---|---|---|---|---|
| **Linear Regression** | **0.9778** | **10.48** | **8.56** | **0.9849 ± 0.0016** |
| Ridge | 0.9779 | 10.46 | 8.54 | 0.9849 ± 0.0015 |
| Lasso | 0.9778 | 10.48 | 8.55 | 0.9849 ± 0.0016 |
| Gradient Boosting | 0.9563 | 14.71 | 11.79 | 0.9597 ± 0.0054 |
| Random Forest | 0.9336 | 18.13 | 13.65 | 0.9412 ± 0.0110 |

> **Key finding:** Linear models outperform ensemble methods on this dataset — a sign that the underlying relationships between features and spending are predominantly **linear**.

---

## Key Findings

### 1. Linear models are best
Linear Regression, Ridge, and Lasso all achieve **R² ≈ 0.978** — explaining 97.8% of variance in yearly spending. This strongly suggests the feature-target relationships are linear, making complex models unnecessary.

### 2. Length of Membership dominates
With a Pearson correlation of **r = 0.809** and Random Forest importance of **70.5%**, membership duration is by far the strongest predictor of customer spending. Long-term customers spend significantly more.

### 3. Time on App matters; Time on Website does not
`Time on App` has a meaningful positive correlation (+0.499) with spending, while `Time on Website` is essentially uncorrelated (r = −0.003). This suggests the company's app is a more effective engagement and conversion channel than its website.

### 4. Cross-validation confirms stability
All models show very low CV standard deviations, confirming no overfitting and reliable generalisation to unseen data.

---

## Business Recommendations

| Insight | Action |
|---|---|
| Membership length drives spending | Invest in loyalty programmes and retention incentives |
| App engagement predicts revenue | Prioritise mobile app improvements over website |
| Website has near-zero impact | Audit website UX — it is not converting engagement to spending |
| Linear model is sufficient | A simple Ridge regression is production-ready and interpretable |

---

## Outputs (10 plots)

| File | Description |
|---|---|
| `01_distributions.png` | Histograms of all 4 features and the target variable |
| `02_pairplot.png` | Pairwise scatter plots across all variables |
| `03_correlation_heatmap.png` | Colour-coded Pearson correlation matrix |
| `04_scatter_plots.png` | Each feature plotted against yearly spending with regression line |
| `05_model_comparison.png` | R² and RMSE bar charts for all 5 models |
| `06_cross_validation.png` | 5-fold CV R² scores with standard deviation error bars |
| `07_actual_vs_predicted.png` | Scatter of actual vs predicted values for all 5 models |
| `08_lr_coefficients.png` | Standardised coefficients from linear regression |
| `09_feature_importance.png` | Feature importances from Random Forest and Gradient Boosting |
| `10_residual_analysis.png` | Residuals vs predicted, residual distribution, Q-Q plot |
