# Development Economics Project
  
## Econometric Analysis of Development and Growth

---

## 1. Project Overview

This project implements a structured empirical analysis of cross-country economic development using:

- **Model 1 (Levels):** determinants of GDP per capita  
- **Model 2 (Growth):** determinants of GDP per capita growth  
- **World Sample (203 countries):** correlation validation and graphical diagnostics  

The workflow follows a full empirical pipeline: data inspection, correlation analysis, scatter diagnostics, regression estimation, and robustness checks.

The entire pipeline is implemented in a single R script located in the `src/` folder.

---

## 2. Repository Structure

```
development/
│
├── data/processed/csv/
│   ├── Model_1_Regression_Table.csv
│   ├── Model_2_Regression_Table.csv
│   ├── Model_1_Regression_Table_World.csv
│   ├── Model_1_Z_Score_Regression_Table.csv
│   ├── Metadata_Table.csv
│   ├── Country_Table.csv
│   ├── Table_Cluster_World.csv
│   ├── Model_1_Caveat_Table.csv
│   └── Model_2_Caveat_Table.csv
│
├── outputs/
│   ├── charts/
│   │   ├── model1_level/
│   │   ├── model2_growth/
│   │   └── world_sample/
│   │
│   └── results_summary/
│       └── results_summary.txt
│
├── src/
│   └── development_economics_analysis.R
│
└── README.md
```

---

## 3. Data Description

### Main datasets

- **Model 1 (22 countries)** — cross-sectional dataset for GDP per capita (levels)  
- **Model 2 (22 countries)** — dataset for GDP per capita growth (average 2000–2023)  
- **World sample (203 countries)** — used for correlations and graphical validation  
- **Z-score dataset** — standardised variables for robustness checks  

### Supporting datasets (documentation)

- Metadata (definitions and sources)  
- Country classification tables  
- Clustering tables (world sample)  
- Model-specific caveats  

---

## 4. Empirical Strategy

Pipeline:

1. Data loading and inspection  
2. Variable renaming and preparation  
3. Correlation matrix computation  
4. Scatter plot diagnostics  
5. Regression estimation (Model 1)  
6. Growth estimation (Model 2)  
7. Robustness checks  

Clustering:

- **Model 1 & Model 2:** region, core growth model, institutional stability  
- **World sample:** region and world cluster classification  

---

## 5. Software and Dependencies

Implemented in **base R**. No external packages required.

Used libraries:

- `stats` → regression models  
- `graphics` → plotting  
- `utils` → data import  

---

## 6. Outputs

### Charts

Saved in:

```
outputs/charts/
```

Subfolders:
- `model1_level/`
- `model2_growth/`
- `world_sample/`

Features:
- country code labels  
- clustering (colour-coded)  
- regression line only when |correlation| ≥ 0.5  
- filtered plots for resource rents (≥10%)  

### Regression Results

Saved in:

```
outputs/results_summary/results_summary.txt
```

Includes:
- correlation matrices  
- regression outputs  
- diagnostics (N, missing values)

---

## 7. Model Specification and Interpretation

### Model 1 — GDP per capita (levels)

Dependent variable: log GDP per capita → coefficients are **semi-elasticities**.

#### Model 1.1 — Baseline

```
log(GDPpc) = α + β₁ Governance + β₂ HCI + β₃ GCF + β₄ Trade + ε
```

Results:
- Governance: positive and significant  
- Human capital: positive (weaker significance)  
- GCF and Trade: not significant  
- R² ≈ 0.57  

**Quantitative interpretation:**
- A +1 increase in governance (WGI scale) is associated with a **large increase in GDP per capita (log scale)**.

**Economic interpretation:**
- Institutions are the strongest predictor of development; human capital matters but is less robust; investment and trade do not explain cross-country income differences in this sample.

#### Model 1.2 — Structure

Manufacturing and services added:
- Coefficients unstable, often not significant, signs change.

**Interpretation:**
- Sectoral composition is not a reliable predictor; multicollinearity likely; structure does not independently explain development.

#### Model 1.3 — Resource Curse

```
+ Resources + (Governance × Resources)
```

Results:
- Resource rents: not robust  
- Interaction: not significant  

**Interpretation:**
- No clear evidence of a resource curse; governance does not significantly modify resource effects.

#### Model 1.4 — Expanded Model

Adds resources and FDI:
- Governance remains significant  
- FDI not significant  
- R² ≈ 0.63  

**Interpretation:**
- Governance remains the only robust determinant; added variables contribute little.

---

### Model 2 — GDP per capita growth

```
Growth = α + β₁ Initial GDP + β₂ Governance + β₃ HCI + β₄ GCF + β₅ Trade + ε
```

Results:
- R² ≈ 0.27  
- Initial GDP: negative (weak convergence)  
- Other variables: not significant  

**Interpretation:**
- Lower-income countries tend to grow slightly faster; growth is not well explained by this specification.

---

## 8. Interpretation of Regression Statistics

- **Coefficients (β):** direction and magnitude  
- **Standard errors:** precision  
- **t-values / p-values:** statistical significance  
- **R-squared:** explanatory power  
- **Adjusted R²:** model comparison  
- **Residual error:** unexplained variation  
- **Observations (N):** sample size  

Key point:
- Model 1 → moderate explanatory power  
- Model 2 → low explanatory power  

---

## 9. Key Findings

- Governance is consistently significant  
- Human capital is important but less stable  
- Structural variables are unreliable  
- Natural resources have weak effects  
- Growth is poorly explained  

**Conclusion:** development is mainly associated with **institutions and human capital**.

---

## 10. Robustness

- Specification comparison  
- Sample variation  
- Outlier inspection (via scatter plots)  
- Standardised regressions (z-scores)  

Result: governance remains robust; other variables are unstable.

---

## 11. Limitations

- Small sample (22 countries)  
- Cross-sectional design (no causality)  
- Data heterogeneity  
- Multicollinearity (structure variables)  
- Weak growth model  

---

## 12. Future Extensions

- Panel data models  
- Fixed/random effects  
- Instrumental variables  
- Non-linear models  
- Larger datasets  

---

## 13. How to Run

Script:

```
src/development_economics_analysis.R
```

Steps:
1. Open the script in R or RStudio  
2. Set:
```
project_dir <- "your/local/path/development"
```
3. Run the script  

Outputs will be generated in:

```
outputs/
```

---

## 14. Final Interpretation

- Institutions and human capital explain income levels  
- Economic structure and resources are not robust drivers  
- Growth dynamics are more complex  

**Income levels are easier to explain than growth processes.**