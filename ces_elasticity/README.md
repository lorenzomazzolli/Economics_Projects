CES ELASTICITY OF SUBSTITUTION PROJECT
=====================================

OVERVIEW
--------
This project estimates the elasticity of substitution between capital and labour using a CES framework across three complementary empirical settings:

1. Cross-country levels (AMECO)
2. Total economy growth panel (OECD)
3. Sectoral growth panel (OECD)

The main objective is to assess whether the estimated elasticity is stable across:
- static vs dynamic specifications
- aggregate vs sectoral data
- full-sample vs trimmed cross-country estimation


INSTALLATION & REQUIREMENTS
---------------------------

Recommended Python version:
- Python 3.10+

Required packages:
- pandas
- numpy
- statsmodels
- matplotlib
- pathlib

Install packages with pip:

pip install pandas numpy statsmodels matplotlib


PROJECT STRUCTURE
-----------------

Root project folder:
- ces_elasticity/

Main folders:
- data/
  Contains input datasets used in the project.

- src/
  Contains the main Python script:
  - ces_elasticity_analysis.py

- outputs/
  Contains all generated outputs.

Output structure:
- outputs/charts/model1/full_sample/
  Full-sample Model 1 charts

- outputs/charts/model1/trimmed_sample/
  Trimmed-sample Model 1 charts

- outputs/charts/model2/scatter/
  Model 2 scatter plots by country

- outputs/charts/model2/lines/
  Model 2 labour-share time series by country

- outputs/charts/model3/scatter/
  Model 3 scatter plots by country-sector

- outputs/charts/model3/lines/
  Model 3 labour-share time series by country-sector

- outputs/results_summary/txt/
  Full regression log:
  - results_summary.txt

- outputs/results_summary/tables/model1/
  Structured CSV outputs for Model 1

- outputs/results_summary/tables/model2/
  Structured CSV outputs for Model 2

- outputs/results_summary/tables/model3/
  Structured CSV outputs for Model 3


METHODOLOGY
-----------

MODEL 1 — CROSS-COUNTRY LEVELS (AMECO)

Specification:
- x = ln(K/H)
- y = ln(sL/sK)

Estimated equation:
- y = alpha + beta*x

Two specifications are estimated:
- Model 1A: full sample
- Model 1B: trimmed sample excluding influential observations based on Cook’s distance


MODEL 2 — TOTAL ECONOMY GROWTH PANEL (OECD)

Transformations:
- dy = Δln(sL/sK)
- dx = gK − gH

Estimated equation:
- dy = alpha + beta*dx + country fixed effects + year fixed effects

Additional features:
- clustered standard errors by country
- unbalanced panel due to missing observations


MODEL 3 — SECTORAL GROWTH PANEL (OECD)

Same transformation logic as Model 2, but with country-sector units.

Estimated equation:
- dy = alpha + beta*dx + country-sector fixed effects + year fixed effects

Additional features:
- clustered standard errors by country-sector
- large heterogeneous panel
- focus charts on selected macro sectors:
  _T, C, BTE, BTNXL, GTNXL


DATA AND SAMPLE
---------------

MODEL 1
- raw observations: 22 countries
- final observations used: 22
- missing values dropped: 0

MODEL 2
- raw observations: 161
- final cleaned observations: 160
- observations used in regression: 153
- countries:
  Canada, France, Germany, Italy, Japan, United Kingdom, United States

Country coverage in regression sample:
- Canada: 22
- France: 22
- Germany: 22
- Italy: 22
- Japan: 22
- United Kingdom: 22
- United States: 21

MODEL 3
- raw observations: 2415
- final cleaned observations: 2254
- observations used in regression: 2153

Main missing-data count in sectoral file:
- Capital_Deepening_Growth_kserhrs: 92
- Gross_Value_Added_gva: 92
- Hours_hrsto: 92
- Total_Labour_Compensation_lctot: 92


RESULTS
-------

MODEL 1A — FULL SAMPLE CROSS-COUNTRY LEVELS

Correlation:
- corr(x, y) = -0.2044

Regression results:
- beta = -0.2051
- sigma = 1 / (1 + beta) = 1.2580
- R-squared = 0.0418
- p-value on beta = 0.329
- N = 22

Interpretation:
- weak negative relationship
- mild apparent substitutability
- very low explanatory power


MODEL 1B — TRIMMED SAMPLE

Influential observations detected via Cook’s distance:
- Ireland
- Norway

Trimmed regression results:
- beta = 0.0875
- sigma = 0.9195
- R-squared = 0.0512
- p-value on beta = 0.381
- N = 20

Interpretation:
- once the two influential outliers are removed, the estimate moves close to Cobb-Douglas
- the cross-country result is not robust


MODEL 2 — TOTAL ECONOMY GROWTH PANEL

Correlation:
- corr(dx, dy) = -0.2460

Regression results:
- beta = -0.4095
- sigma = 1.6936
- R-squared = 0.329
- p-value on beta = 0.150
- N = 153

Interpretation:
- aggregate dynamic evidence points toward substitutability
- however, the slope is not strongly significant
- the result is likely affected by macro shocks and aggregation


MODEL 3 — SECTORAL GROWTH PANEL

Correlation:
- corr(dx, dy) = 0.1404

Regression results:
- beta = 0.5076
- sigma = 0.6633
- R-squared = 0.099
- p-value on beta = 0.050
- N = 2153

Interpretation:
- sectoral evidence points toward complementarity
- the sign flips relative to Model 2
- this suggests that aggregation masks important production heterogeneity


COMPARATIVE SUMMARY
-------------------

Elasticity estimates across models:
- Model 1A (full sample): sigma = 1.2580
- Model 1B (trimmed sample): sigma = 0.9195
- Model 2 (aggregate growth panel): sigma = 1.6936
- Model 3 (sectoral growth panel): sigma = 0.6633

Main message:
- the estimated elasticity is highly sensitive to specification and aggregation level
- cross-country levels are fragile and outlier-sensitive
- aggregate dynamic data suggest substitutability
- sectoral panel data suggest complementarity

Most credible takeaway:
- the sectoral specification provides the strongest evidence
- capital and labour appear more complementary than aggregate models suggest


ECONOMIC INTERPRETATION
-----------------------

This project does not deliver one single stable structural elasticity.

Instead, it shows that:
- full-sample cross-country evidence is distorted by influential countries
- aggregate time-series dynamics can overstate substitutability
- sectoral data provide a very different and more rigid picture of production structure

Overall conclusion:
- Cobb-Douglas remains a useful benchmark at aggregate level
- but sectoral evidence suggests sigma < 1 is more realistic in practice


HOW TO RUN
----------

From the project root folder, run:

python src/ces_elasticity_analysis.py

The script will:
- load all processed CSV inputs
- run all three models
- generate charts
- save txt and csv outputs into the outputs/ folder


OUTPUTS
-------

Main output files:
- outputs/results_summary/txt/results_summary.txt
- charts saved inside outputs/charts/model1/, model2/, and model3/
- structured CSV outputs saved inside outputs/results_summary/tables/

CSV OUTPUTS
-----------

The project exports structured CSV files for each model in:

- outputs/results_summary/tables/model1/
- outputs/results_summary/tables/model2/
- outputs/results_summary/tables/model3/

Model 1 CSV files:
- model1_dataset.csv
  Cleaned full-sample cross-country dataset

- model1_trimmed_dataset.csv
  Cleaned trimmed dataset excluding influential observations

- model1_full_regression.csv
  Full regression table for Model 1A
  (coefficients, standard errors, test statistics, p-values)

- model1_trimmed_regression.csv
  Full regression table for Model 1B

- model1_full_summary.csv
  Compact summary for Model 1A
  (beta, sigma, R-squared, adjusted R-squared, p-value, observations)

- model1_trimmed_summary.csv
  Compact summary for Model 1B

- model1_influence_diagnostics.csv
  Influence diagnostics for Model 1
  (fitted values, residuals, studentized residuals, leverage, Cook’s distance)


Model 2 CSV files:
- panel_dataset_model2.csv
  Cleaned total-economy panel dataset used for transformations and descriptive checks

- model2_regression_sample.csv
  Final sample actually used in the Model 2 regression

- model2_regression.csv
  Full regression table for Model 2
  (coefficients, standard errors, test statistics, p-values)

- model2_summary.csv
  Compact model-level summary
  (beta, sigma, R-squared, adjusted R-squared, p-value, observations)


Model 3 CSV files:
- panel_dataset_model3.csv
  Cleaned country-sector panel dataset used for transformations and descriptive checks

- model3_regression_sample.csv
  Final sample actually used in the Model 3 regression

- model3_regression.csv
  Full regression table for Model 3
  (coefficients, standard errors, test statistics, p-values)

- model3_summary.csv
  Compact model-level summary
  (beta, sigma, R-squared, adjusted R-squared, p-value, observations)


Cross-model CSV file:
- elasticities_summary.csv
  Combined comparison table across models, including:
  - model
  - sample
  - slope variable
  - beta
  - sigma
  - R-squared
  - adjusted R-squared
  - p-value
  - observations


WHY THESE CSV FILES MATTER
--------------------------

These CSV outputs make the project reusable beyond the text summary.

They can be used for:
- Excel analysis and pivot tables
- Power BI dashboards
- replication in R or Stata
- robustness checks
- appendix tables for academic or portfolio use

The text file (results_summary.txt) remains useful for human-readable logs,
but the CSV exports provide the structured outputs needed for serious downstream analysis.


LIMITATIONS
-----------

Current limitations of the project:
- Model 1 uses only one cross-section (2022)
- Model 2 is based on aggregate country-level dynamics and may be driven by macro shocks
- Model 3 is richer, but still depends on sector coverage and data quality
- current CSV exports are still relatively summary-oriented and do not yet fully exploit the richness of the panel outputs


IMPROVEMENTS
------------

Natural next steps for the project:
- export full cleaned model datasets to CSV for downstream Excel / Power BI / econometric analysis
- export full regression tables (coefficients, standard errors, t-stats, p-values)
- export elasticity comparison tables in one combined file
- export residuals, fitted values, leverage, Cook’s distance, and influence diagnostics systematically
- extend the analysis to alternative country groups or broader sector coverage
- test robustness to alternative labour share definitions and additional controls


FINAL TAKEAWAY
--------------

The project suggests that the elasticity of substitution between capital and labour is not a fixed empirical constant.

The strongest result comes from the sectoral panel:
- beta = 0.5076
- sigma = 0.6633

Final interpretation:
capital and labour are often complements in practice, and high substitutability at aggregate level may largely reflect aggregation effects rather than underlying production technology.