
# Productivity, Labour Compensation, Labour Share and Unit Labour Cost  
## Firm-side vs Worker-side Pass-through Analysis (G7)

---

## 1. Project Overview

This project develops a coherent empirical framework to analyse the relationship between labour productivity, labour compensation, real wages, labour shares, and unit labour costs in G7 economies.

The core objective is to quantify the pass-through from productivity to wages, distinguishing between two perspectives:

- Firm-side: labour compensation deflated by output prices (real product wage)  
- Worker-side: real wages in PPP terms (purchasing power)

The analysis is organised into four modules:

1. Firm-side time series  
2. Firm-side cross section  
3. Worker-side time series  
4. Worker-side cross section  

Key design choices:

- Aggregate (total-economy) data only  
- Labour share and ULC analysed within the firm-side framework  
- Worker-side focuses on productivity vs real wages  
- Two labour input definitions:
  - per employee  
  - per employed person  
- Cross-section results are descriptive (G7 only)  

---

## 2. Theoretical Framework

The baseline is a Cobb–Douglas production function under constant returns to scale:

Y = A K^α L^(1−α)

Under standard assumptions:

- Labour share = 1 − α  
- Real product wage = labour productivity × labour share  

Key identity:

Δln(w/p) = Δln(productivity) + Δln(labour share)

Implications:

- If productivity grows faster than wages → labour share declines  
- If wages track productivity → labour share remains stable  

The CES framework (separate project) explains why labour shares vary over time, while this project measures actual pass-through dynamics.

---

## 3. Data and Sample

### Data Sources

- OECD Productivity Database  
- OECD Average Annual Wages (constant prices, constant PPP)

### Sample

- Countries: G7 (United States, United Kingdom, Germany, France, Italy, Japan, Canada)  
- Frequency: Annual  
- Time coverage: variable-specific  
- Structure:
  - Time-series panel (core analysis)  
  - Cross-section (2023 snapshot)

### Firm-side Dataset

Variables (national currency, production-consistent):

- Gross Value Added (nominal and real)  
- GVA per hour (real and nominal)  
- Labour compensation (total and per hour)  
- Unit labour cost (ULC) index  
- Labour share  

Used to construct:

- productivity  
- output-price inflation  
- real product wage  
- labour share  
- ULC  

### Worker-side Dataset

Variables (PPP, purchasing power):

- Real GVA (PPP-based)  
- Employment and employees  
- Real wages (PPP, constant prices)

Two productivity definitions:

- per employee  
- per employed person  

### Data Pipeline

All data are:

- cleaned and harmonised  
- transformed into log growth rates  
- used to construct indices  
- assembled into panel datasets  

Pipeline implemented in:

productivity_labourcompensation_distribution/src/product_labcomp_distr_analysis.py

Outputs include:

- processed datasets  
- regression outputs  
- charts  
- summary tables  

---

## 4. Methodology

### 4.1 Firm-side Time Series

Key transformations:

g_prod_real = Δln(real GVA per hour)  
g_prod_nom = Δln(nominal GVA per hour)  

Output prices:

g_P_output = g_prod_nom − g_prod_real  

Real product wage:

g_w_real_output = g_LC_nom − g_P_output  

Labour share:

LS = Labour Compensation / Nominal GVA  

Gap:

Gap_firm = ln(Productivity index / Real product wage index)

---

### 4.2 Firm-side Regressions

Model 1A:

Δln(w_real_output) = α_i + λ_t + β Δln(productivity)

Result:  
β ≈ 0.44 → partial pass-through  

Model 1B:

Δln(w_real_output) = Δln(productivity) + Δln(labour share)

This is an accounting identity, not a behavioural regression.

Model 1C:

Δln(ULC) = α_i + λ_t + β Δln(productivity)

Result:  
β ≈ -0.56 → productivity reduces unit labour costs  

---

### 4.3 Worker-side Time Series

Δln(w_real_PPP) = α_i + λ_t + γ Δln(productivity)

Results:

- γ ≈ 0.36 (employee-based)  
- γ ≈ 0.37 (employed-based)  

---

### 4.4 Cross Section (2023)

Firm-side:

ln(labour compensation per hour) ~ ln(productivity per hour)

Worker-side:

ln(real wage PPP) ~ ln(productivity)

---

## 5. Index Construction

- Within-country: first common year  
- Cross-country: first common year across countries  

This ensures comparability and correct cumulative interpretation.

---

## 6. Results

### Firm-side

- β ≈ 0.44 → partial pass-through  
- Productivity grows faster than wages → labour share declines  

### Worker-side

- γ ≈ 0.36–0.37 → weaker pass-through  

### Interpretation

Firm-side pass-through is higher than worker-side.

Main drivers:

- labour share  
- price wedge (output vs consumption prices)

---

## 7. Comparative Summary (G7)

- Italy and Japan show the largest productivity–wage gaps  
- United Kingdom shows near full pass-through  
- United States shows high productivity with moderate divergence  
- Germany and France are intermediate cases  

---

## 8. Economic Interpretation

Three mechanisms explain the results:

1. Incomplete pass-through  
2. Labour share adjustment  
3. Output vs consumption price wedge  

Firm-side reflects cost dynamics, while worker-side reflects purchasing power.

---

## 9. Final Takeaway

Productivity growth does not automatically translate into real wages.

- Firms retain part of productivity gains  
- Workers capture even less  
- The gap is persistent and structural  

---

## 10. Limitations

- G7-only sample  
- Small cross-section  
- No sectoral decomposition  
- Labour share approximation issues  
- PPP vs output price mismatch  
- Reduced-form regressions (not causal)

---

## 11. Project Structure

project/

data/ → Excel and CSV input datasets  
docs/ → methodology document  
outputs/charts/ → firm, worker, comparative charts  
outputs/csv/ → processed, metadata, summary datasets  
outputs/txt/ → regression outputs and logs  
src/product_labcomp_distr_analysis → main pipeline  

---

## 12. Installation and Requirements

pip install pandas numpy matplotlib statsmodels pathlib

---

## 13. How to Run

python src/product_labcomp_distr_analysis

Outputs generated:

- charts  
- CSV datasets  
- summary tables  
- textual results  

---

## 14. Extensions

- Extend to OECD panel  
- Add sector-level analysis  
- Integrate CES elasticity results  
- Analyse price wedge dynamics  

---