# US Monetary Policy & Business Cycle Analysis

## Overview

This project provides an empirical analysis of key macroeconomic relationships in the United States using time-series data.

The goal is to replicate and interpret core macroeconomic frameworks:

- Sahm Rule (recession indicator)
- Beveridge Curve (labour market dynamics)
- Okun’s Law (output–unemployment relationship)
- Phillips Curve (inflation dynamics)
- Taylor Rule (monetary policy benchmark)

The analysis combines data processing, statistical estimation, and graphical interpretation.

---

## Project Structure

- `data/processed/csv/` → input datasets  
- `outputs/` → charts and results  
- `main.py` → full pipeline  
- `results_summary.txt` → regression outputs  

---

## Methodology

- Data handled using pandas
- Regressions estimated with OLS (statsmodels)
- Visual analysis through scatter plots and time-series charts

---

## Results

### 1. Sahm Rule

The Sahm Rule measures increases in unemployment relative to its recent minimum.

A recession signal occurs when the indicator exceeds 0.5.

**Findings:**
- Correctly identifies major recessions
- Large spike during COVID (2020)
- Latest values remain below recession threshold

**Interpretation:**
The US economy is currently not in recession based on labour market data.

---

### 2. Beveridge Curve

The Beveridge Curve shows the relationship between unemployment and job vacancies.

**Findings:**
- Negative relationship between unemployment and vacancies
- Stronger correlation when excluding COVID period
- Evidence of structural shift after 2020

**Interpretation:**
COVID temporarily disrupted the labour market matching process.  
After the pandemic, the relationship becomes more stable again.

---

### Labour Market Tightness

Measured as vacancies divided by unemployment.

**Findings:**
- Peak in 2022 indicates very tight labour market
- Sharp drop in 2020 followed by strong recovery

**Interpretation:**
The labour market experienced an unusually strong post-pandemic recovery and is now normalizing.

---

### 3. Okun’s Law

Links economic activity to unemployment.

**Findings:**
- Strong negative relationship between output gap and unemployment gap
- Results are consistent across specifications

**Interpretation:**
Economic growth plays a key role in reducing unemployment.

---

### 4. Phillips Curve

Explains inflation as a function of expectations and labour market slack.

**Findings:**
- Expected inflation is significant
- Unemployment gap has a negative effect
- Overall explanatory power is limited

**Interpretation:**
Inflation is influenced by labour market conditions, but expectations are more important.  
This is consistent with modern macroeconomic theory.

---

### 5. Taylor Rule

Provides a benchmark for interest rate policy.

**Findings:**
- Implied rates are often higher than actual Fed policy rates
- Large deviations during:
  - 2008 financial crisis
  - COVID period
- Core inflation produces smoother results than headline inflation

**Important Note:**
Inflation is measured using annualized quarterly data (SAAR), which can increase volatility and lead to higher implied rates.

**Interpretation:**
The Federal Reserve does not strictly follow a Taylor rule, especially during extreme economic conditions.

---

## Conclusion

The analysis confirms standard macroeconomic relationships:

- Labour market indicators effectively signal economic cycles
- Output and unemployment are strongly connected
- Inflation dynamics are increasingly driven by expectations
- Monetary policy deviates from simple rules when facing shocks

---

## Outputs

All regression results and diagnostics are available in:

- `outputs/`
- `results_summary.txt`

## How to Run

Run the script from the project folder:

```bash
python src/analysis.py

## Requirements

This project uses Python with the following libraries:
- pandas
- numpy
- matplotlib
- statsmodels
- pathlib