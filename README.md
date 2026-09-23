# E-commerce SQL Analytics & Sales Forecasting

A comprehensive data analytics, star-schema data modeling, and forecasting pipeline designed to analyze retail transaction data, derive core business metrics using advanced SQL queries, and project short-term future sales revenue using time-series forecasting.

---

## Executive Summary & Architecture

This repository showcases an end-to-end data analytics workflow:
1. **Data Modeling & Schema Design**: Implemented a **Star Schema** architecture featuring explicit Fact tables (`orders`, `order_items`) and Dimension tables (`customers`, `products`).
2. **Advanced SQL Analytics**: Utilizes window functions (`DENSE_RANK()`, `SUM() OVER()`), complex multi-table joins, and time-series aggregations to extract customer purchasing trends, top-performing product categories, and daily sales metrics.
3. **Sales Forecasting**: Integrates an **Exponential Smoothing** time-series forecasting model (`run_forecasting.py`) to generate short-term (7-day ahead) daily revenue projections.
4. **Deployment Scope**: CLI and Notebook-based analytics execution pipeline.

---

## Technical Stack

* **Database**: SQLite
* **Data Processing & Analytics**: Python 3, Pandas, NumPy, SQL / SQLite3
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
pip install pandas matplotlib numpy

# 3. Execute Data Pipeline
# Step A: Initialize Database & Seed Data
python setup_db.py

# Step B: Run Advanced SQL Analytics & Export Visualizations
python run_analysis.py

# Step C: Execute Sales Forecasting Model
python run_forecasting.py