import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# 1. Load Daily Sales Metrics
df = pd.read_csv('daily_sales_metrics.csv')

date_col = [col for col in df.columns if 'date' in col.lower() or 'day' in col.lower()][0]
revenue_col = [col for col in df.columns if 'revenue' in col.lower() or 'sales' in col.lower() or 'total' in col.lower()][0]

df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(by=date_col)
df.set_index(date_col, inplace=True)

# Ensure regular daily frequency
df = df.asfreq('D').ffill()

# 2. Train / Test Split for Model Evaluation
test_days = 7
train_df = df.iloc[:-test_days]
test_df = df.iloc[-test_days:]

# 3. Model Training: Holt-Winters Exponential Smoothing
model = ExponentialSmoothing(
    train_df[revenue_col], 
    trend='add', 
    seasonal=None, 
    initialization_method="estimated"
).fit()

# Evaluate on Test Set
test_predictions = model.forecast(test_days)
mae = mean_absolute_error(test_df[revenue_col], test_predictions)
rmse = root_mean_squared_error(test_df[revenue_col], test_predictions)

print(f"=== Model Evaluation Metrics ===")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")

# 4. Refit Model on Full Dataset & Forecast 7 Days into Future
full_model = ExponentialSmoothing(
    df[revenue_col], 
    trend='add', 
    seasonal=None, 
    initialization_method="estimated"
).fit()

future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=7, freq='D')
future_forecast = full_model.forecast(7)

# 5. Visualization: Actual vs Fitted vs Future Forecast
plt.figure(figsize=(12, 6))

# Historical Actuals
plt.plot(df.index, df[revenue_col], label='Historical Actual Revenue', color='#1f77b4', linewidth=2)

# Test Predictions
plt.plot(test_df.index, test_predictions, label='Validation Predictions (Test)', color='#2ca02c', linestyle='--', linewidth=2)

# Future Forecast
plt.plot(future_dates, future_forecast, label='7-Day Future Forecast', color='#d62728', linestyle='--', linewidth=2.5, marker='o')

plt.title('E-commerce Daily Revenue: Historical vs Validation vs 7-Day Forecast', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=11)
plt.ylabel('Revenue ($)', fontsize=11)
plt.legend(loc='upper left')
plt.grid(True, linestyle=':', alpha=0.6)

# Annotate Evaluation Metrics on Plot
plt.annotate(
    f'Validation Metrics:\nMAE: ${mae:.2f}\nRMSE: ${rmse:.2f}', 
    xy=(0.75, 0.15), 
    xycoords='axes fraction',
    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=1)
)

plt.tight_layout()
plt.savefig('sales_forecast.png', dpi=300)
print("Updated sales forecast image generated successfully: sales_forecast.png")