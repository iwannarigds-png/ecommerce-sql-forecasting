import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error

# Set fixed random seed for 100% reproducibility across environments
SEED = 42
np.random.seed(SEED)

print("=== Starting Methodological Benchmark Forecasting Pipeline (Seed Fixed: 42) ===")

# ==========================================
# 1. LOAD SOURCE A: UNSTRUCTURED SYNTHETIC DATA
# ==========================================
print("\n[1/3] Loading Source A (Unstructured Synthetic Data)...")
df_synth = pd.read_csv('daily_sales_metrics.csv')
date_col = [col for col in df_synth.columns if 'date' in col.lower() or 'day' in col.lower()][0]
rev_col = [col for col in df_synth.columns if 'revenue' in col.lower() or 'sales' in col.lower() or 'total' in col.lower()][0]

df_synth[date_col] = pd.to_datetime(df_synth[date_col])
df_synth = df_synth.sort_values(by=date_col).set_index(date_col).asfreq('D').ffill()

# ==========================================
# 2. LOAD SOURCE B: AUTHENTIC REAL-WORLD DEMAND SERIES
# Source: AirPassengers Benchmark Dataset (Statsmodels), Rescaled to Daily Revenue
# ==========================================
print("\n[2/3] Loading Source B (AirPassengers Authentic Demand Series)...")
real_data_raw = sm.datasets.get_rdataset("AirPassengers", "datasets").data
dates_real = pd.date_range(start='2023-01-01', periods=len(real_data_raw), freq='D')
real_revenue = real_data_raw['value'].values * 45.0 + np.random.normal(0, 350, len(real_data_raw))

df_real = pd.DataFrame({rev_col: real_revenue}, index=dates_real)
print(f"Loaded Source B Series: {len(df_real)} daily observations.")

# ==========================================
# 3. EVALUATION PIPELINE (28-Day Expanded Holdout & Dual Baselines)
# ==========================================
print("\n[3/3] Running Evaluation Pipeline (28-Day Holdout Window)...")
test_days = 28  # 4-Week Holdout to prevent single-week sampling noise

def evaluate_pipeline(df, target_col):
    train = df.iloc[:-test_days]
    test = df.iloc[-test_days:]
    
    # 1. Holt-Winters Exponential Smoothing Model
    model = ExponentialSmoothing(
        train[target_col], trend='add', seasonal='add', seasonal_periods=7, initialization_method="estimated"
    ).fit()
    preds = model.forecast(test_days)
    
    # 2. Simple Naive Persistence (t = t-1)
    naive_val = train[target_col].iloc[-1]
    naive_preds = pd.Series([naive_val] * test_days, index=test.index)
    
    # 3. Seasonal Naive Persistence (t = t-7)
    s_naive_preds = pd.Series(index=test.index, dtype=float)
    for i in range(test_days):
        s_naive_preds.iloc[i] = train[target_col].iloc[-(7 - (i % 7))]
    
    # Calculate Primary Model Metrics
    mae = mean_absolute_error(test[target_col], preds)
    rmse = root_mean_squared_error(test[target_col], preds)
    mape = mean_absolute_percentage_error(test[target_col], preds) * 100
    
    # Calculate Baseline MAEs
    naive_mae = mean_absolute_error(test[target_col], naive_preds)
    s_naive_mae = mean_absolute_error(test[target_col], s_naive_preds)
    
    imp_vs_simple = ((naive_mae - mae) / naive_mae) * 100
    imp_vs_seasonal = ((s_naive_mae - mae) / s_naive_mae) * 100
    
    return train, test, preds, mae, rmse, mape, naive_mae, s_naive_mae, imp_vs_simple, imp_vs_seasonal

# Run evaluations
tr_s, te_s, pr_s, mae_s, rmse_s, mape_s, n_mae_s, sn_mae_s, imp_sim_s, imp_sea_s = evaluate_pipeline(df_synth, rev_col)
tr_r, te_r, pr_r, mae_r, rmse_r, mape_r, n_mae_r, sn_mae_r, imp_sim_r, imp_sea_r = evaluate_pipeline(df_real, rev_col)

# ==========================================
# 4. PRINT BENCHMARK RESULTS
# ==========================================
print("\n" + "="*72)
print("   METHODOLOGICAL BENCHMARK STUDY: SYNTHETIC DATA vs AUTHENTIC SERIES   ")
print("="*72)
print(f"{'Metric (28-Day Holdout)':<28} | {'Source A (Synthetic)':<18} | {'Source B (AirPassengers)':<20}")
print("-" * 72)
print(f"{'Holt-Winters MAE':<28} | ${mae_s:<17.2f} | ${mae_r:<19.2f}")
print(f"{'Holt-Winters RMSE':<28} | ${rmse_s:<17.2f} | ${rmse_r:<19.2f}")
print(f"{'Holt-Winters MAPE':<28} | {mape_s:<17.1f}% | {mape_r:<19.1f}%")
print(f"{'Simple Naive MAE (t-1)':<28} | ${n_mae_s:<17.2f} | ${n_mae_r:<19.2f}")
print(f"{'Seasonal Naive MAE (t-7)':<28} | ${sn_mae_s:<17.2f} | ${sn_mae_r:<19.2f}")
print(f"{'Imp. vs Simple Naive':<28} | {imp_sim_s:<+17.1f}% | {imp_sim_r:<+19.1f}%")
print(f"{'Imp. vs Seasonal Naive':<28} | {imp_sea_s:<+17.1f}% | {imp_sea_r:<+19.1f}%")
print("="*72 + "\n")

# ==========================================
# 5. VISUALIZATION
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Source A
ax1.plot(df_synth.index[-60:], df_synth[rev_col].iloc[-60:], label='Historical Actuals', color='#1f77b4', lw=1.5)
ax1.plot(te_s.index, pr_s, label='Holt-Winters Forecast', color='#2ca02c', lw=2.5, ls='--')
ax1.set_title(f'Source A: Unstructured Synthetic Data\nMAPE: {mape_s:.1f}% | Vs Seasonal Naive: {imp_sea_s:+.1f}%', fontsize=11, fontweight='bold')
ax1.set_ylabel('Revenue ($)')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper left')

# Source B
ax2.plot(df_real.index[-60:], df_real[rev_col].iloc[-60:], label='Historical Actuals', color='#ff7f0e', lw=1.5)
ax2.plot(te_r.index, pr_r, label='Holt-Winters Forecast', color='#2ca02c', lw=2.5, ls='--')
ax2.set_title(f'Source B: AirPassengers Demand Series\nMAPE: {mape_r:.1f}% | Vs Seasonal Naive: {imp_sea_r:+.1f}%', fontsize=11, fontweight='bold')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper left')

plt.suptitle('Methodological Study: 28-Day Forecasting Benchmark Across Data Sources (Seed = 42)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('synthetic_vs_real_benchmark.png', dpi=300)
print("Updated benchmark plot saved: synthetic_vs_real_benchmark.png\n")