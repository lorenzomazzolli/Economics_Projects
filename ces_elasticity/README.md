CES Elasticity of Substitution (Capital vs Labour)

--------------------------------------------------

Overview

This project provides an empirical analysis of the elasticity of substitution between capital and labour using cross-country and panel data.

The goal is to test the empirical validity of the CES framework across different specifications and levels of aggregation.

The analysis is structured around three complementary models:

- Cross-country level relationship (AMECO)
- Aggregate growth panel (OECD, G7)
- Sectoral growth panel (OECD)

The project combines data construction, regression analysis, and graphical interpretation.

--------------------------------------------------

Project Structure

- data/processed/csv/ → input datasets  
- outputs/ → charts and results  
- ces_elasticity_analysis.py → full pipeline  
- results_summary.txt → regression outputs  

--------------------------------------------------

Methodology

- Data handled using pandas  
- Regressions estimated with OLS (statsmodels)  
- Fixed effects used in panel models  
- Variables expressed in logs and log-differences  

Core specification:

Model 1 (levels):
ln(sL/sK) ~ ln(K/H)

Model 2–3 (growth):
Δln(sL/sK) ~ Δln(K/H)

--------------------------------------------------

Results

1. Cross-country Model (Levels – AMECO)

This model estimates the CES relationship using a 2022 cross-section.

Findings:
- Weak negative relationship in full sample  
- Very low explanatory power  
- Results not statistically significant  

After removing influential observations (Ireland, Norway):

- Relationship becomes slightly positive  
- Elasticity moves close to Cobb–Douglas (σ ≈ 1)  
- Still not statistically significant  

Interpretation:

The cross-country estimate is not robust and highly sensitive to outliers.  
This model should be interpreted as a diagnostic benchmark rather than a reliable estimate.

--------------------------------------------------

2. Aggregate Growth Model (G7 – OECD)

This model uses time variation within countries.

Findings:
- Negative relationship between capital deepening and labour share growth  
- Implied elasticity σ > 1 (substitutability)  
- Coefficient not statistically significant  

Interpretation:

There is weak evidence of substitutability, but the relationship is noisy and unstable.  
Macroeconomic shocks (e.g. 2008, COVID) dominate the signal.

--------------------------------------------------

3. Sectoral Growth Model (OECD)

This model extends the analysis to sector-level data.

Findings:
- Positive relationship between Δln(K/H) and Δln(sL/sK)  
- Implied elasticity σ < 1 (complementarity)  
- Borderline statistical significance  

Interpretation:

Once sectoral heterogeneity is introduced, the sign of the relationship changes.  
This suggests that aggregation masks important within-economy dynamics.

--------------------------------------------------

Conclusion

The three models do not deliver a single stable estimate of the elasticity of substitution.

Instead, the results show:

- Strong sensitivity to outliers (Model 1)  
- Weak and unstable aggregate relationship (Model 2)  
- Different dynamics at sector level (Model 3)  

Key takeaway:

Empirical estimates of the CES elasticity are highly sensitive to data structure, aggregation level, and model specification.

This project should therefore be interpreted as a robustness analysis rather than a precise estimation exercise.

--------------------------------------------------

Outputs

All regression results and diagnostics are available in:

- outputs/
- results_summary.txt

--------------------------------------------------

How to Run

Run the script from the project folder:

python ces_elasticity_analysis.py

--------------------------------------------------

Requirements

This project uses Python with the following libraries:

- pandas  
- numpy  
- matplotlib  
- statsmodels  
- pathlib  
