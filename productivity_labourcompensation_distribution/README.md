# Productivity – Labour Compensation – Labour Share – ULC (G7 Analysis)

## Overview

This project analyses the relationship between labour productivity, wages, labour share, and unit labour costs (ULC) across G7 economies.

The goal is to quantify the pass-through from productivity to wages, distinguishing between:

- Firm-side perspective → real product wage, labour share, cost structure  
- Worker-side perspective → real wages (PPP), purchasing power  

The analysis combines:
- panel time-series regressions (main results)
- cross-section analysis (supporting evidence)
- accounting-based decomposition checks

---

## Project Structure

project_root/

data/  
│   ├── raw/  
│   └── processed/  

outputs/  
│   ├── charts/  
│   │   ├── firm_side/  
│   │   │   ├── time_series/  
│   │   │   └── cross_section/  
│   │   ├── worker_side/  
│   │   │   ├── time_series/  
│   │   │   └── cross_section/  
│   │   └── comparative/  
│   │  
│   ├── csv/  
│   └── txt/  

src/  
│   ├── main.py  
│   ├── helpers.py  
│   └── config.py  

README.md

---

## Requirements

Install required Python packages:

pip install pandas numpy matplotlib seaborn statsmodels pathlib

Optional:

pip install jupyter notebook


---

## Data

The project uses pre-cleaned CSV datasets containing:

Firm-side:
- GVA (nominal and real)
- Labour compensation
- Unit labour costs (ULC)

Worker-side:
- Real wages (PPP, FTE)
- Employment / employees
- Productivity measures

Place datasets in:

data/raw/

---

## Run the Project

python src/product_labcomp_distr_analysis.py

The pipeline will:
1. Load and clean data  
2. Compute transformations  
3. Estimate regressions  
4. Generate charts  
5. Export outputs  

---

## Data Processing Pipeline

All variables are transformed using log-differences.

Productivity:
g_prod_real = Δln(GVA per hour, real)

Output price:
g_Pout = g_prod_nom – g_prod_real

Real product wage:
g_w_real_output = g_lc_nom – g_Pout

Labour share:
LS = Labour compensation / GVA

ULC:
ULC ≈ LC / Productivity

---

## Indices (Base 100)

Indices are constructed using consistent rebasing rules:

- Within-country comparisons → first common available year  
- Cross-country comparisons → first common year across countries  

---

## Methodology

### Firm-side models

Model 1A – Pass-through  
g_w_real_output ~ g_prod_real + FE(country) + FE(year)

Measures how productivity translates into real product wages.

---

Model 1B – Labour share identity  
g_w_real_output ≈ g_prod_real + Δln(labour_share)

This is an accounting identity (Cobb–Douglas benchmark).  
Used only as a consistency check.

---

Model 1C – ULC robustness  
g_ulc_nom ~ g_prod_real

Measures cost pressure dynamics.

---

### Worker-side models

g_wage_real_ppp ~ g_prod_employee  
g_wage_real_ppp ~ g_prod_employed  

Measures pass-through to purchasing power.

---

### Cross-section

- Log-log regressions (G7, latest year)
- Used as supporting evidence only

---

## Results

### Firm-side pass-through (Model 1A)

Coefficient: 0.436  
Highly significant  

Interpretation:
Productivity growth is only partially transmitted to real product wages.

---

### Labour share identity (Model 1B)

Coefficients:
- Productivity: 1.000  
- Labour share: 1.000  
R² = 1.000  

Interpretation:
Pure accounting identity, not an empirical result.

---

### ULC (Model 1C)

ULC growth ≈ wage growth – productivity growth  

Interpretation:
Acts as a consistency and cost-pressure check.

---

### Worker-side pass-through

Employee-based: 0.356  
Employed-based: 0.370  

Interpretation:
Lower than firm-side → incomplete transmission to purchasing power.

---

### Cross-section (G7)

Firm-side: 0.981  
Worker-side: weaker  

Interpretation:
Consistent with theory but limited by small sample.

---

## Gap Analysis

gap = ln(Productivity index / Wage index)

Compared to:
- ln(labour_share)

Result:
Correlation ≈ 0.23  

Interpretation:
Theoretical link exists but not perfectly matched empirically.

---

## Outputs

CSV:
- Processed datasets

TXT:
- Regression summaries
- Correlation matrices
- Consistency checks

Charts:

Firm-side:
- Productivity vs wage indices  
- ULC index  
- Labour share (dual axis)  
- Gap  
- Scatter plots  

Worker-side:
- PPP wage vs productivity  
- Gap (employee vs employed)  

Comparative:
- Productivity index  
- Wage index  
- ULC index  
- Labour share index  
- Gap  

---

## Theoretical Interpretation

Cobb–Douglas benchmark:

Real wage ≈ productivity  
Labour share constant  

Empirical findings:
- Pass-through < 1  
- Labour share varies  
- Firm-side ≠ worker-side  

CES interpretation:
- Time-varying labour share  
- Capital–labour substitution  
- Explains incomplete pass-through  

---

## Limitations

- Small cross-section (G7)  
- PPP vs national accounts mismatch  
- Labour share sensitive to self-employment  
- Index construction depends on base year  

---

## Next Steps

- Extend to OECD panel  
- Sector-level analysis  
- Integrate CES estimation  

---

## Summary

Productivity growth is not fully transmitted to wages.  
The gap is persistent and systematic.  
Worker-side effects are weaker than firm-side.  

Distribution and labour share dynamics are central.
