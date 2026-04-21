# US Macroeconomic Framework: Monetary Policy and Business Cycle Dynamics

---

## 1. Project Objective

This project develops an empirical macroeconomic framework for the United States to analyse the interaction between monetary policy, labour market conditions, and the business cycle.

The analysis integrates five core macroeconomic relationships:

- Sahm Rule (recession indicator)
- Beveridge Curve (labour market matching)
- Okun’s Law (output–unemployment relationship)
- Phillips Curve (inflation dynamics)
- Taylor Rule (monetary policy benchmark)

---

## 2. Installation and Requirements

This project requires Python 3 and the following libraries:

- pandas  
- numpy  
- matplotlib  
- statsmodels
- pathlib  

Install dependencies with:

    pip install pandas numpy matplotlib statsmodels pathlib

---

## 3. Project Structure

The project follows a clear input → processing → output pipeline:

    us_macro_analysis/
    ├── data/                     # Input data (processed Excel and CSV files)
    │   └── processed/csv/        # Final datasets imported into Python
    │
    ├── docs/                     # Methodology and supporting documentation
    │
    ├── outputs/                  # Generated outputs
    │   ├── charts/               # All charts (PNG format)
    │   └── results_summary/      # Regression outputs (TXT format)
    │
    ├── src/                      # Main analysis pipeline
    │   └── us_macro_analysis.py  # Full workflow: data → models → outputs
    │
    └── README.md

**Workflow logic:**
- Data → prepared in Excel/CSV and stored in `data/`
- Pipeline → executed via Python script in `src/`
- Outputs → automatically generated in `outputs/`

---

## 4. Data and Sample

The dataset is constructed using FRED time series and includes:

- Real GDP and Potential GDP  
- GDP Growth (SAAR)  
- PCE Inflation (Headline and Core, SAAR)  
- Unemployment Rate (UNRATE)  
- Natural Rate of Unemployment (NROU)  
- Inflation Expectations (5Y5Y Forward)  
- Job Openings Rate (JOLTS)  
- Federal Funds Rate  

Sample periods vary across models depending on data availability:

- Sahm Rule → ~1950–present  
- Okun’s Law / Phillips Curve / Taylor Rule → ~1960–present  
- Beveridge Curve → ~2000–present  

Derived variables:

- Output Gap  
- Output Growth Gap  
- Unemployment Gap  
- ΔUnemployment  
- Labour Market Tightness = Vacancies / Unemployment  

All variables are aligned into a consistent time-series framework and transformed into gaps, growth rates, or ratios depending on the model specification.

---

## 5. Methodology

The framework distinguishes between:

### Regression-based models
- Okun’s Law (levels and differences)
- Phillips Curve
- Beveridge Curve

### Indicator and benchmark frameworks
- Sahm Rule
- Taylor Rule (implemented as an implied benchmark, not estimated)

All regressions are estimated using OLS with a constant term.

The Beveridge Curve is estimated linearly, while a quadratic fit is used for visualization purposes.

---

## 6. Results

The empirical results provide evidence on the strength, stability, and limitations of each macroeconomic relationship.

### 6.1 Sahm Rule

- Latest value (Feb 2026): **0.267**
- Threshold: **0.5 → no recession signal**
- Peak: **9.43 (June 2020)**

---

### 6.2 Beveridge Curve

- Correlation (full sample): **-0.59**
- Correlation (ex-COVID): **-0.74**
- OLS coefficient: **-0.91 (significant)**

Quadratic fit parameters:
- a = 0.105  
- b = -1.824  
- c = 10.252  

---

### 6.3 Labour Market Tightness

- Peak: **~2.0 (2022)**
- Latest: **~0.9–1.1 (2025)**

---

### 6.4 Okun’s Law

Level Specification:
- Correlation: **-0.89**
- β: **-0.62**
- R²: **0.78**

Difference Specification:
- Correlation: **-0.74**
- β: **-0.12**
- R²: **0.55**

---

### 6.5 Phillips Curve

- R²: **0.20**
- Expected inflation coefficient: **+2.31**
- Unemployment gap coefficient: **-0.32**

---

### 6.6 Taylor Rule

- Implied rates significantly exceed actual policy rates in several periods  

Example (2022):
- Implied (core): **~8–9%**
- Actual: **<3% initially**

---

## 7. Economic Interpretation

- Labour market indicators show strong post-COVID normalisation  
- Okun’s Law confirms a stable relationship between output and unemployment  
- The Phillips Curve is statistically valid but relatively weak  
- Monetary policy deviates from simple rule-based benchmarks during large shocks  

---

## 8. Comparative Summary

Relationship | Strength | Key Insight  
--- | --- | ---  
Sahm Rule | High | Reliable recession indicator  
Beveridge Curve | Medium | Structural shift post-COVID  
Okun’s Law | High | Strong output–labour link  
Phillips Curve | Low | Expectations dominate  
Taylor Rule | Medium | Policy deviates in crises  

---

## 9. Outputs

The pipeline generates:

- Charts → outputs/charts/  
- Results → outputs/results_summary/results_summary.txt  

---

## 10. How to Run

Run the analysis from the project folder:

    python src/us_macro_analysis.py

---

## 11. Limitations

- No dynamic specification (lags not included)  
- No robust standard errors (HAC/Newey-West)  
- Static Phillips Curve  
- Beveridge Curve estimated in reduced form  

---

## 12. Possible Improvements

- Introduce dynamic models (VAR, ARDL)  
- Estimate time-varying Phillips Curve  
- Include alternative expectation measures  
- Structural modelling of labour market matching  
- Estimate Taylor Rule instead of calibrating it  

---

## 13. Final Takeaways

- The US labour market has normalised after COVID  
- Okun’s Law remains strongly supported by the data  
- Inflation dynamics are increasingly driven by expectations  
- Monetary policy cannot be fully captured by simple rules  