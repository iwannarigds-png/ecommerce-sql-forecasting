import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load daily sales metrics generated from analysis
df = pd.read_csv('daily_sales_metrics.csv')

# Ensure date column is properly parsed
date_col = [col for col in df.columns if 'date' in col.lower() or 'day' in col.lower()][0]
revenue_col = [col for col in df.columns if 'revenue' in col.lower() or 'sales' in col.lower() or 'total' in col.lower()][0]

df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(by=date_col)
df.set_index(date_col, inplace=True)

# 2. Exponential Smoothing Forecast (7-day ahead forecast)
alpha = 0.3
df['forecast'] = df[revenue_col].ewm(alpha=alpha, adjust=False).mean()

# Project next 7 days
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7)
last_forecast_value = df['forecast'].iloc[-1]

future_df = pd.DataFrame({
    revenue_col: [np.nan] * 7,
    'forecast': [last_forecast_value] * 7
}, index=future_dates)

combined_df = pd.concat([df[[revenue_col, 'forecast']], future_df])

# 3. Plot Historical vs Forecasted Revenue
plt.figure(figsize=(12, 6))
plt.plot(combined_df.index[:-7], combined_df[revenue_col][:-7], label='Historical Revenue', color='#1f77b4', linewidth=2)
plt.plot(combined_df.index[-8:], combined_df['forecast'][-8:], label='7-Day Sales Forecast', color='#ff7f0e', linestyle='--', linewidth=2.5)

plt.title('Daily Revenue & 7-Day Sales Forecast', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Revenue ($)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

plt.savefig('sales_forecast.png', dpi=300)
print("Sales forecast generated successfully: sales_forecast.png")