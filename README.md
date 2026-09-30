# E-commerce SQL Analytics & Sales Forecasting

A comprehensive data analytics, star-schema data modeling, and time-series forecasting pipeline designed to analyze retail transaction data, derive core business insights via advanced SQL, and project short-term future sales revenue.

---

## Executive Summary & Architecture

This repository showcases an end-to-end data engineering & analytics workflow:
1. **Data Modeling & Schema Design**: Designed a **Star Schema** architecture featuring explicit Fact tables (`orders`, `order_items`) and Dimension tables (`customers`, `products`).
2. **Advanced SQL Analytics**: Utilizes window functions (`DENSE_RANK()`, `SUM() OVER()`), multi-table joins, and time-series aggregations to extract customer purchasing trends, top-performing product categories, and daily metrics.
3. **Forecasting & Validation**: Implements a **Holt-Winters Exponential Smoothing** time-series model (`run_forecasting.py`) trained on daily revenue data. Evaluated against a naive persistence benchmark using MAE, RMSE, and MAPE.
4. **Deployment & Project Scope**: Designed intentionally as a CLI & Notebook-based analytical pipeline for integration into automated workflows (no Streamlit deployment).

---

## Key Business Findings

Based on SQL analytics executed across the Star Schema database:
* **Top Product Category**: Electronics & Accessories account for the highest proportion of total revenue, generating over **35%** of overall gross sales.
* **Customer Retention & Order Value**: Repeat buyers contribute significantly higher Average Order Value (AOV) compared to first-time shoppers.
* **Daily Sales Velocity**: Revenue exhibits clear daily fluctuations, highlighting key high-volume transaction days during mid-week periods.

---

## Forecasting Approach & Performance Results

To evaluate the predictive model, a holdout validation strategy (7-day test set) was applied against historical sales data.

### Model Specification
* **Algorithm**: Holt-Winters Exponential Smoothing (`statsmodels.tsa.holtwinters`)
* **Components**: Additive Trend (`trend='add'`), fitted on daily aggregated revenue ($).
* **Evaluation Baseline**: Evaluated against a Naive Persistence Baseline (predicting tomorrow = last known daily revenue).

### Metrics Summary
* **Mean Absolute Error (MAE)**: **$1,282.46**
* **Root Mean Squared Error (RMSE)**: **$1,739.43**
* **Mean Absolute Percentage Error (MAPE)**: **~35.0%**

> **Methodological Insight**: Time-series models evaluated on synthetic data provide a baseline benchmark. Evaluating against a naive baseline demonstrates how smoothing algorithms handle noisy variance versus real-world seasonality signals.

---

## Technical Stack

* **Database**: SQLite
* **Data Processing & SQL**: Python 3, Pandas, NumPy, SQL / SQLite3
* **Machine Learning & Time Series**: Statsmodels, Scikit-Learn
* **Data Visualization**: Matplotlib
* **Version Control**: Git & GitHub

---

## How to Run

Follow these steps to set up the environment, populate the database, execute SQL analytics, and generate sales forecasts locally:

```bash
# 1. Clone the Repository
git clone [https://github.com/iwannarigds-png/ecommerce-sql-forecasting.git](https://github.com/iwannarigds-png/ecommerce-sql-forecasting.git)
cd ecommerce-sql-forecasting

# 2. Set Up Virtual Environment & Dependencies
python3 -m venv venv
source venv/bin/activate
pip install pandas matplotlib numpy statsmodels scikit-learn

# 3. Execute Data Pipeline
python setup_db.py         # Step A: Initialize Database & Seed Data
python run_analysis.py     # Step B: Run Advanced SQL Analytics & Export Visualizations
python run_forecasting.py  # Step C: Execute Sales Forecasting Model & Evaluate