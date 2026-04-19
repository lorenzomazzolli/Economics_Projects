# Economics Projects

This repository contains a set of applied economics and macroeconomic analysis projects, focusing on data analysis, econometrics, and policy-relevant insights.

## Projects

### 1. Development

Cross-country panel analysis of economic development, including governance, human capital, capital accumulation, and trade openness.

- Data sources: World Bank (WDI, WGI)
- Methods: Panel regressions, correlation analysis
- Folder: `/development`

---

### 2. Productivity, Labour Compensation, Labour Share, and Unit Labour Cost

Analysis of the relationship between productivity, labour compensation, labour share, and unit labour cost.

- Focus: pass-through from productivity to labour compensation
- Data sources: OECD, AMECO
- Methods: time-series analysis, decomposition, regressions
- Folder: `/productivity_labourcomp_labourshare_ulc`

---

### 3. US Monetary Policy and Business Cycle

Empirical analysis of core macroeconomic relationships using US data.

- Models:
  - Taylor Rule
  - Phillips Curve
  - Okun’s Law
  - Beveridge Curve
  - Sahm Rule
- Data source: FRED
- Methods: time-series regression, macro modeling
- Folder: `/us_macro_analysis`

---

### 4. CES Elasticity of Substitution

Estimation of the elasticity of substitution between capital and labour using a CES production framework.

- Approach: log-linearized CES derived from first-order conditions
- Data sources: OECD, AMECO
- Methods: regression analysis
- Folder: `/ces_elasticity`

---

### 5. Wages, Income, Labour Share, and Unit Labour Cost

Analysis of the relationship between wages and household disposable income, alongside labour share and labour cost dynamics and decomposition. 

- Focus: Analysis of wages, household disposable income, labour share, and unit labour costs, combining wage–income comparisons with distributional and cost-competitiveness frameworks.
- Data sources: OECD
- Methods: time-series analysis, decomposition
- Folder: /wages_income_labourshare_ulc

---

## Repository Structure 

Each project follows a consistent structure:

project/
├── data/
│   ├── raw/        # original data (Excel, source files)
│   └── processed/  # cleaned datasets (CSV for analysis)
├── src/            # Python scripts
├── outputs/        # charts, tables, results
└── docs/           # methodology and notes

---

## Objective

The objective of this repository is to build a portfolio of applied economic analysis projects combining:

- data handling and wrangling
- econometric modeling
- macroeconomic interpretation
- policy-relevant insights
