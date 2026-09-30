import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error

# 1. Load Daily Sales Metrics
df = pd.read_csv('daily_sales_metrics.csv')

date_col = [col for col in df.columns if 'date' in col.lower() or 'day' in col.lower()][0]
revenue_col = [col for col in df.columns if 'revenue' in col.lower() or 'sales' in col.lower() or 'total' in col.lower()][0]

df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(by=date_col)
df.set_index(date_col, inplace=True)

# Ensure regular daily frequency
df = df.asfreq('D').ffill()

# 2. Train / Test Split (7-day holdout evaluation)
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

# Predictions on Test Set
test_predictions = model.forecast(test_days)

# Naive Baseline (Persistence: Forecast = Last Observed Value)
naive_value = train_df[revenue_col].iloc[-1]
naive_predictions = pd.Series([naive_value] * test_days, index=test_df.index)

# Metrics Calculation
mae = mean_absolute_error(test_df[revenue_col], test_predictions)
rmse = root_mean_squared_error(test_df[revenue_col], test_predictions)
mape = mean_absolute_percentage_error(test_df[revenue_col], test_predictions) * 100

naive_mae = mean_absolute_error(test_df[revenue_col], naive_predictions)
mae_improvement = ((naive_mae - mae) / naive_mae) * 100 if naive_mae != 0 else 0.0

mean_daily_revenue = test_df[revenue_col].mean()

print(f"=== Model Evaluation Metrics (7-Day Holdout) ===")
print(f"Average Daily Revenue: ${mean_daily_revenue:.2f}")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Naive Baseline MAE: ${naive_mae:.2f}")
print(f"Improvement vs Naive Baseline: {mae_improvement:+.1f}%")

# 4. Refit Model on Full Dataset & Forecast 7 Days into Future
full_model = ExponentialSmoothing(
    df[revenue_col], 
    trend='add', 
    seasonal=None, 
    initialization_method="estimated"
).fit()

future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=7, freq='D')
future_forecast = full_model.forecast(7)

# 5. Visualization: Actual vs Test Validation vs 7-Day Future Forecast
plt.figure(figsize=(12, 6))

# Historical Actuals
plt.plot(df.index, df[revenue_col], label='Historical Revenue', color='#1f77b4', linewidth=1.5)

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
    f'Validation Metrics (7-Day Test):\nMAE: ${mae:.2f} (MAPE: {mape:.1f}%)\nRMSE: ${rmse:.2f}', 
    xy=(0.68, 0.15), 
    xycoords='axes fraction',
    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=1)
)

plt.tight_layout()
plt.savefig('sales_forecast.png', dpi=300)
print("\nUpdated sales forecast image saved: sales_forecast.png")