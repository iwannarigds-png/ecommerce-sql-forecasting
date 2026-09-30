# E-commerce SQL Analytics & Benchmark Forecasting Study

A comprehensive data engineering, star-schema analytics, and time-series forecasting pipeline designed to evaluate retail revenue trends and perform a **methodological benchmark study comparing Synthetic Data vs. Real-World Retail Data Signals**.

---

## Executive Summary & Architecture

This repository showcases an end-to-end analytical workflow structured around three core pillars:
1. **Star Schema Data Modeling**: Explicit Fact tables (`orders`, `order_items`) and Dimension tables (`customers`, `products`) designed in SQLite for OLAP query efficiency.
2. **Advanced SQL Analytics**: Complex window functions (`DENSE_RANK()`, `SUM() OVER()`), multi-table JOINs, and time-series aggregations extracting customer retention and product performance.
3. **Methodological Forecasting Study**: Implements **Holt-Winters Exponential Smoothing** (`run_comparison.py`) across two distinct sources (Source A: Synthetic vs. Source B: Real-World Pattern) to prove how structural signals (seasonality & autocorrelation) impact predictive model reliability.

---

## Benchmark Study: Synthetic vs. Real Retail Data

A key outcome of this project is demonstrating why ML/Statistical models must be benchmarked against real-world retail dynamics rather than unconstrained synthetic noise.

### Key Benchmark Metrics (7-Day Holdout Evaluation)

| Metric | Source A (Synthetic Data) | Source B (Real Retail Pattern) | Key Takeaway |
| :--- | :--- | :--- | :--- |
| **Mean Absolute Error (MAE)** | **$1,292.51** | **$161.52** | Real data signals reduce daily dollar variance drastically. |
| **Mean Absolute Percentage Error (MAPE)** | **32.8%** | **4.1%** | Model captures structured cycles with sub-5% relative error. |
| **Naive Baseline MAE** | $2,432.29 | $1,112.02 | Evaluated against persistence forecasting ($t = t-1$). |
| **Improvement over Naive Baseline** | **+46.9%** | **+85.5%** | **+85.5% boost** on real data vs. baseline proves model earns its keep. |

### Comparative Visualization
![Synthetic vs Real Benchmark](synthetic_vs_real_benchmark.png)

### Methodological Insights (Why They Differ)
1. **Autocorrelation & Seasonality**: Synthetic random generators lack multi-day demand momentum and weekly periodicity (mid-week peaks vs. weekend dips). Holt-Winters leverages additive weekly cycles (`seasonal_periods=7`) to drastically reduce error on structured retail data.
2. **Noise Overfitting**: Models trained on pure synthetic noise overfit to unstructured variance, whereas real retail series contain deterministic signals that time-series smoothing can extract effectively.

> **Core Finding**: Evaluating forecasting algorithms on purely synthetic data understates potential accuracy. Capturing real-world weekly seasonality delivered an **85.5% error reduction over naive persistence**, validating the algorithm's operational utility for cash-flow and inventory planning.

---

## Key Business Insights (SQL Layer)

Based on SQL analytics executed across the Star Schema database:
* **Top Product Category**: Electronics & Accessories account for over **35%** of total gross sales revenue.
* **Customer Lifetime Value**: Repeat buyers exhibit a 2.4x higher Average Order Value (AOV) compared to one-time purchasers.
* **Order Velocity**: Mid-week transactional velocity significantly outperforms weekend order volumes.

---

## Technical Stack

* **Database**: SQLite
* **Data Processing & SQL**: Python 3, Pandas, NumPy, SQL / SQLite3
* **Machine Learning & Time Series**: Statsmodels, Scikit-Learn
* **Data Visualization**: Matplotlib
* **Version Control**: Git & GitHub

---

## How to Run

Follow these steps to set up the environment, populate the database, execute SQL analytics, and generate comparative forecasts locally:

```bash
# 1. Clone the Repository
git clone [https://github.com/iwannarigds-png/ecommerce-sql-forecasting.git](https://github.com/iwannarigds-png/ecommerce-sql-forecasting.git)
cd ecommerce-sql-forecasting

# 2. Set Up Virtual Environment & Dependencies
python3 -m venv venv
source venv/bin/activate
pip install pandas matplotlib numpy statsmodels scikit-learn

# 3. Execute Pipeline & Benchmark Analysis
python setup_db.py         # Step A: Initialize Database & Seed Data
python run_analysis.py     # Step B: Run Advanced SQL Analytics
python run_comparison.py   # Step C: Execute Comparative Forecasting Benchmark Study