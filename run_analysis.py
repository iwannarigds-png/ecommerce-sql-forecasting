
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Connect to the SQLite database
conn = sqlite3.connect("retail_data.db")

# ---------------------------------------------------------
# Query 1: Time-Series Aggregations & Window Functions
# ---------------------------------------------------------
query_timeseries = """
WITH DailySales AS (
    SELECT 
        order_date,
        COUNT(DISTINCT order_id) AS total_orders,
        SUM(quantity) AS total_units_sold,
        ROUND(SUM(quantity * unit_price), 2) AS daily_revenue
    FROM transactions
    GROUP BY order_date
),
SalesMetrics AS (
    SELECT 
        order_date,
        daily_revenue,
        ROUND(AVG(daily_revenue) OVER (
            ORDER BY order_date 
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ), 2) AS rolling_avg_7d,
        LAG(daily_revenue, 1) OVER (ORDER BY order_date) AS prev_day_revenue
    FROM DailySales
)
SELECT 
    order_date,
    daily_revenue,
    rolling_avg_7d,
    prev_day_revenue,
    ROUND(((daily_revenue - prev_day_revenue) / prev_day_revenue) * 100, 2) AS dod_growth_pct
FROM SalesMetrics
ORDER BY order_date;
"""

# ---------------------------------------------------------
# Query 2: Relational JOINs, Aggregations & Category Ranking
# ---------------------------------------------------------
query_category = """
WITH CategorySales AS (
    SELECT 
        p.category,
        COUNT(DISTINCT t.order_id) AS total_orders,
        SUM(t.quantity) AS total_units_sold,
        ROUND(SUM(t.quantity * t.unit_price), 2) AS total_revenue,
        ROUND(AVG(t.quantity * t.unit_price), 2) AS avg_order_value
    FROM transactions t
    JOIN products p ON t.product_id = p.product_id
    GROUP BY p.category
    HAVING SUM(t.quantity * t.unit_price) > 10000
)
SELECT 
    category,
    total_orders,
    total_units_sold,
    total_revenue,
    avg_order_value,
    DENSE_RANK() OVER (ORDER BY total_revenue DESC) AS revenue_rank
FROM CategorySales
ORDER BY revenue_rank;
"""

# Execute queries
df_timeseries = pd.read_sql_query(query_timeseries, conn)
df_category = pd.read_sql_query(query_category, conn)
conn.close()

# Export results to CSV for documentation
df_timeseries.to_csv("daily_sales_metrics.csv", index=False)
df_category.to_csv("category_performance.csv", index=False)
print("\n Results exported to CSV files successfully!")

# ---------------------------------------------------------
# Plotting Time-Series Trends
# ---------------------------------------------------------
df_timeseries['order_date'] = pd.to_datetime(df_timeseries['order_date'])

plt.figure(figsize=(12, 6))
plt.plot(df_timeseries['order_date'], df_timeseries['daily_revenue'], alpha=0.4, label='Daily Revenue ($)', color='gray')
plt.plot(df_timeseries['order_date'], df_timeseries['rolling_avg_7d'], label='7-Day Moving Average ($)', color='blue', linewidth=2)

plt.title("E-commerce Daily Revenue Trend & 7-Day Moving Average", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Revenue ($)", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Save chart image
plt.savefig("revenue_trend.png", dpi=300, bbox_inches='tight')
print(" Chart saved as 'revenue_trend.png'!")