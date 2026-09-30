# E-commerce SQL Analytics & Methodological Benchmark Forecasting Study

A comprehensive data engineering, star-schema analytics, and time-series forecasting study designed to evaluate retail revenue trends and perform a **rigorous benchmark comparison between Synthetic Schema Data and Authentic Real-World E-Commerce Transactions**.

---

## Executive Summary & Architecture

This repository showcases an end-to-end analytical workflow structured around three core pillars:
1. **Star Schema Data Modeling**: Explicit Fact tables (`orders`, `order_items`) and Dimension tables (`customers`, `products`) designed in SQLite for OLAP query efficiency.
2. **Advanced SQL Analytics**: Analytical queries utilizing window functions (`DENSE_RANK()`, `SUM() OVER()`), multi-table JOINs, and time-series aggregations for exploratory data analysis on synthetic database schema.
3. **Methodological Forecasting Benchmark Study**: Implements **Holt-Winters Exponential Smoothing** (`run_comparison.py`) across two distinct e-commerce datasets to evaluate predictive performance against dual naive baselines (Simple Persistence $t-1$ and Seasonal Persistence $t-7$) over an extended 28-day holdout window.

---

## Benchmark Study: Synthetic Schema vs. UCI Real E-Commerce Dataset

A primary objective of this study is demonstrating model validation behavior across unstructured synthetic noise and authentic, transactional e-commerce revenue streams.

### Methodological Benchmark Metrics (28-Day Holdout Evaluation | Fixed Seed: 42)

| Metric (28-Day Holdout Window) | Source A (Synthetic E-Commerce Schema) | Source B (UCI Online Retail Real Dataset) | Operational Takeaway |
| :--- | :--- | :--- | :--- |
| **Holt-Winters MAE** | **$1,244.10** | **$13,732.06** | Captures operational forecast error across daily revenue scales. |
| **Holt-Winters RMSE** | **$1,553.45** | **$28,524.25** | Penalizes high-magnitude residual outliers in transaction spikes. |
| **Holt-Winters MAPE** | **44.4%** | **27.2%** | Demonstrates stabilized relative accuracy on authentic series. |
| **Simple Naive MAE ($t-1$)** | $1,454.46 | $16,780.76 | Evaluated against standard daily persistence baseline. |
| **Seasonal Naive MAE ($t-7$)** | $1,767.89 | $21,510.29 | Evaluated against weekly seasonal persistence baseline. |
| **Imp. vs Simple Naive** | **+14.5%** | **+18.2%** | Holt-Winters demonstrates real value add over simple persistence. |
| **Imp. vs Seasonal Naive** | **+29.6%** | **+36.2%** | Captures complex multi-day weekly trends better than $t-7$ heuristics. |

*Note: Data transformation pipelines and model initialization protocols use `np.random.seed(42)` to guarantee 100% deterministic reproducibility.*

### Data Sources & Lineage
* **Source A (Synthetic Schema)**: Generated via relational SQLite schema mirroring transactional e-commerce metrics.
* **Source B (UCI Online Retail Dataset)**: Authentic UK-based e-commerce transactional logs aggregated to daily total revenue ($Quantity \times UnitPrice$).

### Comparative Benchmark Visualization
![Synthetic vs Real Benchmark](synthetic_vs_real_benchmark.png)

### Key Analytical Findings
1. **Predictive Value Add on Authentic Data**: On real-world e-commerce transaction logs, Holt-Winters Exponential Smoothing outperforms both Simple Naive persistence (+18.2%) and Seasonal Naive persistence (+36.2%), confirming model efficacy on true trended retail data.
2. **Audit Governance Imperative**: Benchmarking models against both simple ($t-1$) and seasonal ($t-7$) baselines across a 28-day backtest window provides the necessary rigor to prevent over-reliance on unvalidated heuristics.

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