# E-commerce SQL Analytics & Sales Forecasting

A comprehensive data analytics, star-schema data modeling, and time-series forecasting pipeline designed to analyze retail transaction data, derive business metrics via advanced SQL, and project short-term future sales revenue.

---

## Executive Summary & Architecture

This repository showcases an end-to-end data engineering & analytics workflow:
1. **Data Modeling & Schema Design**: Designed a **Star Schema** architecture featuring explicit Fact tables (`orders`, `order_items`) and Dimension tables (`customers`, `products`).
2. **Advanced SQL Analytics**: Utilizes window functions (`DENSE_RANK()`, `SUM() OVER()`), multi-table joins, and time-series aggregations to extract customer purchasing trends, top-performing product categories, and daily metrics.
3. **Forecasting Approach & Results**: Implements a **Holt-Winters Exponential Smoothing** time-series model (`run_forecasting.py`) trained on daily revenue data.
4. **Deployment & Project Scope**: Designed intentionally as a CLI & Notebook-based analytical pipeline for reproduction and execution in automated workflows (no Streamlit deployment).

---

## Forecasting Approach & Performance Results

To evaluate the predictive model, a holdout validation strategy (Train/Test split) was applied using the last 7 days of historical sales data.

* **Model**: Holt-Winters Exponential Smoothing (`statsmodels.tsa.holtwinters`)
* **Target Metric**: Daily Total Revenue ($)
* **Mean Absolute Error (MAE)**: **$1,282.46** (average daily dollar deviation on holdout test set)
* **Root Mean Squared Error (RMSE)**: **$1,739.43** (penalizes larger variance errors)

> **Key Takeaway**: The model captures baseline revenue trends effectively, providing a reliable short-term 7-day sales projection for inventory and cash-flow planning.

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