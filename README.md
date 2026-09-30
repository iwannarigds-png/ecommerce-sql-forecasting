# E-commerce SQL Analytics & Methodological Benchmark Forecasting Study

A comprehensive data engineering, star-schema analytics, and time-series forecasting study designed to evaluate retail revenue trends and perform a **rigorous benchmark comparison between Synthetic Data and Real-World Series**.

---

## Executive Summary & Architecture

This repository showcases an end-to-end analytical workflow structured around three core pillars:
1. **Star Schema Data Modeling**: Explicit Fact tables (`orders`, `order_items`) and Dimension tables (`customers`, `products`) designed in SQLite for OLAP query efficiency.
2. **Advanced SQL Analytics**: Analytical queries utilizing window functions (`DENSE_RANK()`, `SUM() OVER()`), multi-table JOINs, and time-series aggregations for exploratory data analysis on synthetic data.
3. **Methodological Forecasting Benchmark Study**: Implements **Holt-Winters Exponential Smoothing** (`run_comparison.py`) across two distinct data sources (Source A: Unstructured Synthetic Data vs. Source B: Authentic Real-World Demand Data) to evaluate predictive model performance and baseline persistence.

---

## Benchmark Study: Synthetic Data vs. Real-World Series

A primary objective of this study is demonstrating how model evaluation varies significantly between unconstrained synthetic noise and real-world trended time series.

### Methodological Benchmark Metrics (7-Day Holdout Evaluation)

| Metric | Source A (Synthetic Data) | Source B (Authentic Real Data) | Operational Takeaway |
| :--- | :--- | :--- | :--- |
| **Holdout MAE** | **$1,292.51** | **$3,527.48** | Real data dynamics reflect higher scale and variance. |
| **Holdout RMSE** | **$1,826.82** | **$4,037.17** | Captures magnitude of penalization for extreme residuals. |
| **Holdout MAPE** | **32.8%** | **15.2%** | Substantially lower relative percentage error on real series. |
| **Naive Baseline MAE** | $2,432.29 | $3,366.92 | Evaluated against persistence forecasting ($t = t-1$). |
| **Baseline Improvement** | **+46.9%** | **-4.8%** | Highlighted the strength of simple naive baselines on volatile real series. |

### Comparative Benchmark Visualization
![Synthetic vs Real Benchmark](synthetic_vs_real_benchmark.png)

### Key Analytical Findings
1. **Relative Error Discrepancy**: Synthetic unstructured noise results in a high MAPE (32.8%), whereas real-world trended data allows the Holt-Winters model to stabilize relative percentage error at 15.2%.
2. **Naive Baseline Dynamics**: On volatile real-world time series, naive persistence ($t = t-1$) remains a highly competitive baseline (-4.8% delta), proving that complex smoothing models must be continuously audited against simple heuristics.

---

## Technical Stack

* **Database**: SQLite
* **Data Processing & Analytics**: Python 3, Pandas, NumPy, SQL / SQLite3
* **Machine Learning & Time Series**: Statsmodels, Scikit-Learn
* **Visualization**: Matplotlib
* **Version Control**: Git & GitHub

---

## How to Run

Follow these steps to set up the environment, populate the database, execute SQL analytics, and run the comparative benchmark study locally:

```bash
# 1. Clone the Repository
git clone [https://github.com/iwannarigds-png/ecommerce-sql-forecasting.git](https://github.com/iwannarigds-png/ecommerce-sql-forecasting.git)
cd ecommerce-sql-forecasting

# 2. Set Up Virtual Environment & Dependencies
python3 -m venv venv
source venv/bin/activate
pip install pandas matplotlib numpy statsmodels scikit-learn

# 3. Execute Pipeline & Benchmark Analysis
python setup_db.py         # Step A: Initialize Database & Seed Synthetic Data
python run_analysis.py     # Step B: Run Advanced SQL Analytics
python run_comparison.py   # Step C: Execute Methodological Benchmark Forecasting Study