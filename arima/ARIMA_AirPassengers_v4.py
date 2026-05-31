# ARIMA Time Series Forecasting - Proof of Concept
# https://www.kaggle.com/code/sunaysawant/air-passengers-time-series-arima/notebook
# Dataset: Air Passengers
# This script is designed to be adapted to use different transformations and p, d, q values to bulid the model:
# - The values of p, d, q can be user defined in line 636
# - Ensure the correct single line is uncommentened in the block of lines 646 - 649 - to apply the relevant tranformation
# - Ensure the correct rows are uncommented in section 668- 677 to inverse the corrections applied

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import time
import warnings
warnings.filterwarnings('ignore')

# Start timer
t0 = time.time()  # Add at start of process

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================
print("\nSTEP 1: LOADING AIR PASSENGERS DATASET")

# Load the dataset
# 'AirPassengers.csv' from the Kaggle link above
df = pd.read_csv('AirPassengers.csv')
print(f"\nDataset shape: {df.shape}")
print("\nSample data:")
print(df)

# Convert to time series
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
df.columns = ['Passengers']

# Split into train (80%) and test (20%)
train_size = int(len(df) * 0.8)
train_data = df[:train_size]
test_data = df[train_size:]

print(f"\nTrain set size: {len(train_data)} observations")
print(f"Test set size: {len(test_data)} observations")

# Visualize the original data
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data['Passengers'], label='Training Data', linewidth=2)
plt.plot(test_data.index, test_data['Passengers'], label='Test Data', linewidth=2, color='orange')
plt.title('Air Passengers Time Series - Train/Test Split', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Passengers', fontsize=12)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('arima_data_split.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# STEP 1a: SEASONAL DECOMPOSITION
# ============================================================================
print("\nSTEP 1a: SEASONAL DECOMPOSITION")

decomposition = seasonal_decompose(train_data['Passengers'], model='multiplicative', period=12)

fig, axes = plt.subplots(4, 1, figsize=(12, 12))
decomposition.observed.plot(ax=axes[0], color='steelblue', linewidth=1.5)
axes[0].set_title('Observed', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Passengers', fontsize=11)

decomposition.trend.plot(ax=axes[1], color='darkorange', linewidth=1.5)
axes[1].set_title('Trend Component', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Trend', fontsize=11)

decomposition.seasonal.plot(ax=axes[2], color='seagreen', linewidth=1.5)
axes[2].set_title('Seasonal Component (Period = 12 Months)', fontsize=13, fontweight='bold')
axes[2].set_ylabel('Seasonality', fontsize=11)

decomposition.resid.plot(ax=axes[3], color='purple', linewidth=1.5)
axes[3].set_title('Residual Component', fontsize=13, fontweight='bold')
axes[3].set_ylabel('Residual', fontsize=11)

for ax in axes:
    ax.set_xlabel('Date', fontsize=11)
    ax.grid(True, alpha=0.3)

plt.suptitle('Multiplicative Seasonal Decomposition - Air Passengers (Training Data)',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('arima_seasonal_decomposition.png', dpi=150, bbox_inches='tight')
plt.show()

# Extract and report seasonal indices
seasonal_indices = decomposition.seasonal[:12]
print(f"\nMonthly Seasonal Indices (multiplicative):")
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
for month, idx in zip(month_names, seasonal_indices):
    direction = 'above' if idx > 1 else 'below'
    print(f"  {month}: {idx:.4f}  ({abs(1-idx)*100:.1f}% {direction} trend)")


# ============================================================================
# STEP 2: STATIONARITY TEST (BEFORE DIFFERENCING)
# ============================================================================
print("\nSTEP 2: ANALYSING STATIONARITY - ORIGINAL DATA")

# Calculate mean and variance for original data
original_mean = train_data['Passengers'].mean()
original_variance = train_data['Passengers'].var()
original_std = train_data['Passengers'].std()

print(f"\nOriginal Data Statistics:")
print(f"Mean: {original_mean:.2f}")
print(f"Standard Deviation: {original_std:.2f}")
print(f"Variance: {original_variance:.2f}")

# Augmented Dickey-Fuller Test
adf_result = adfuller(train_data['Passengers'])
print(f"\nAugmented Dickey-Fuller Test (Original Data):")
print(f"ADF Statistic: {adf_result[0]:.3f}")
print(f"p-value: {adf_result[1]:.3f}")
print(f"Critical Values:")
for key, value in adf_result[4].items():
    print(f" {key}: {value:.3f}")

print('ADF test conclusion:')
if adf_result[1] > 0.05:
    print(f"\nData is NON-STATIONARY (p-value > 0.05)")
    print(f"Differencing is required!")
else:
    print(f"\nData is STATIONARY (p-value <= 0.05)")

# Visualize rolling mean and variance
plt.figure(figsize=(12, 6))
rolling_mean = train_data['Passengers'].rolling(window=12).mean()
rolling_var = train_data['Passengers'].rolling(window=12).var()
plt.plot(train_data.index, train_data['Passengers'], label='Original Data', alpha=0.7)
plt.plot(rolling_mean.index, rolling_mean, label='Rolling Mean (12 months)', linewidth=2, color='red')
plt.title('Original Data - Rolling Mean (Non-Stationary)', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Passengers', fontsize=12)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('arima_train_mean.png', dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(rolling_var.index, rolling_var, label='Rolling Variance (12 months)', linewidth=2, color='green')
plt.title('Original Data - Rolling Variance (Non-Stationary)', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Variance', fontsize=12)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('arima_train_variance.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# STEP 3: INTEGRATED / DIFFERENCING (d in ARIMA)
# ============================================================================
print("\nSTEP 3: APPLYING DIFFERENCING TO ACHIEVE STATIONARITY")

# Apply first-order differencing
train_diff = train_data['Passengers'].diff().dropna()

# Calculate mean and variance after differencing
diff_mean = train_diff.mean()
diff_std = train_diff.std()
diff_variance = train_diff.var()

print(f"\nDifferenced Data Statistics:")
print(f"Mean: {diff_mean:.2f}")
print(f"Standard Deviation: {diff_std:.2f}")
print(f"Variance: {diff_variance:.2f}")

print(f"\nChange in Statistics:")
print(f"Mean change: {original_mean:.2f} → {diff_mean:.2f} (Δ = {abs(original_mean - diff_mean):.2f})")
print(f"Variance change: {original_variance:.2f} → {diff_variance:.2f} (Δ = {abs(original_variance - diff_variance):.2f})")

# ADF test on differenced data
adf_diff = adfuller(train_diff)
print(f"\nAugmented Dickey-Fuller Test (Differenced Data):")
print(f"ADF Statistic: {adf_diff[0]:.4f}")
print(f"p-value: {adf_diff[1]:.4f}")

if adf_diff[1] <= 0.05:
    print(f"\nData is now STATIONARY (p-value <= 0.05)")
    print(f"Differencing order (d) = 1")
else:
    print(f"\nData may need additional differencing")

# Visualize differenced data
plt.figure(figsize=(12, 6))
plt.plot(train_diff.index, train_diff, label='Differenced Data', color='purple', linewidth=1.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
plt.title('Differenced Data (First Order)', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Differenced Passengers', fontsize=12)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('arima_diff.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# STEP 3_KPSS: KPSS TEST - COMPLEMENTARY STATIONARITY CHECK
# ============================================================================
print("\nSTEP 3: KPSS TEST - COMPLEMENTARY STATIONARITY CHECK")
print("Note: ADF null = non-stationary; KPSS null = stationary.")
print("      Both tests together give stronger evidence than either alone.")

# KPSS on original data
kpss_orig = kpss(train_data['Passengers'], regression='ct', nlags='auto')
print(f"\nKPSS Test (Original Data):")
print(f"  KPSS Statistic: {kpss_orig[0]:.4f}")
print(f"  p-value: {kpss_orig[1]:.4f}")
print(f"  Critical Values: {kpss_orig[3]}")
if kpss_orig[1] < 0.05:
    print("  KPSS Conclusion: Reject H₀ → Evidence of NON-STATIONARITY")
else:
    print("  KPSS Conclusion: Cannot reject H₀ → No evidence against stationarity")

# KPSS on first-order differenced data
kpss_diff = kpss(train_diff, regression='c', nlags='auto')
print(f"\nKPSS Test (First-Order Differenced Data):")
print(f"  KPSS Statistic: {kpss_diff[0]:.4f}")
print(f"  p-value: {kpss_diff[1]:.4f}")
if kpss_diff[1] < 0.05:
    print("  KPSS Conclusion: Reject H₀ → Evidence of NON-STATIONARITY (further differencing may be needed)")
else:
    print("  KPSS Conclusion: Cannot reject H₀ → Evidence of STATIONARITY")

# Second-order differencing
train_diff2 = train_diff.diff().dropna()
adf_diff2 = adfuller(train_diff2)
kpss_diff2 = kpss(train_diff2, regression='c', nlags='auto')

print(f"\nSecond-Order Differencing:")
print(f"  ADF p-value: {adf_diff2[1]:.4f}  {'✓ Stationary' if adf_diff2[1] <= 0.05 else '✗ Non-stationary'}")
print(f"  KPSS p-value: {kpss_diff2[1]:.4f}  {'✓ Stationary' if kpss_diff2[1] >= 0.05 else '✗ Non-stationary'}")
print(f"\nThis supports the choice of d=2 for the optimal ARIMA model.")

# Comparison summary
print(f"\nDifferencing Summary:")
print(f"{'Series':<40} {'ADF p-val':>10} {'ADF Result':>16} {'KPSS p-val':>11} {'KPSS Result':>16}")
print("-" * 95)
rows = [
    ("Original", adf_result[1], kpss_orig[1]),
    ("First-order diff (d=1)", adf_diff[1], kpss_diff[1]),
    ("Second-order diff (d=2)", adf_diff2[1], kpss_diff2[1]),
]
for name, adf_p, kpss_p in rows:
    adf_label = "Stationary" if adf_p <= 0.05 else "Non-stationary"
    kpss_label = "Stationary" if kpss_p >= 0.05 else "Non-stationary"
    print(f"{name:<40} {adf_p:>10.4f} {adf_label:>16} {kpss_p:>11.4f} {kpss_label:>16}")


# ============================================================================
# STEP 3a: INTEGRATED / DIFFERENCING (d in ARIMA) - Log of original data
# ============================================================================
print("\nSTEP 3a: APPLYING LOG TRANSFORMATION OF DATA PLUS DIFFERENCING TO ACHIEVE STATIONARITY")

# Take Logarithm of Original Data
train_data['Passengers_Log'] = np.log(train_data['Passengers'])

# Apply first-order differencing
train_log_diff = train_data['Passengers_Log'].diff().dropna()

# Calculate mean and variance after differencing
log_diff_mean = train_log_diff.mean()
log_diff_std = train_log_diff.std()
log_diff_variance = train_log_diff.var()

print(f"\nDifferenced Data Statistics - Log Transformation:")
print(f"Mean: {log_diff_mean:.2f}")
print(f"Standard Deviation: {log_diff_std:.2f}")
print(f"Variance: {log_diff_variance:.2f}")

print(f"\nChange in Statistics - Log Transformation:")
print(f"Mean change: {original_mean:.2f} → {log_diff_mean:.2f} (Δ = {abs(original_mean - log_diff_mean):.2f})")
print(f"Variance change: {original_variance:.2f} → {log_diff_variance:.2f} (Δ = {abs(original_variance - log_diff_variance):.2f})")

# ADF test on differenced data
adf_log_diff = adfuller(train_log_diff)
print(f"\nAugmented Dickey-Fuller Test (Differenced Data):")
print(f"ADF Statistic: {adf_log_diff[0]:.4f}")
print(f"p-value: {adf_log_diff[1]:.4f}")

if adf_log_diff[1] <= 0.05:
    print(f"\nData is now STATIONARY (p-value <= 0.05)")
    print(f"Differencing order (d) = 1")
else:
    print(f"\nData may need additional differencing")

# Visualize differenced data
plt.figure(figsize=(12, 6))
plt.plot(train_log_diff.index, train_log_diff, label='Differenced Data', color='purple', linewidth=1.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
plt.title('Differenced Data (First Order) - Log Transform', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Differenced Passengers', fontsize=12)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('arima_diff_log.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================================
# STEP 3b: INTEGRATED / DIFFERENCING (d in ARIMA) - Square-root of original data
# ============================================================================
print("\nSTEP 3b: APPLYING SQUARE ROOT TRANSFORMATION OF DATA PLUS DIFFERENCING TO ACHIEVE STATIONARITY")

# Take Logarithm of Original Data
train_data['Passengers_Sqrt'] = np.sqrt(train_data['Passengers'])

# Apply first-order differencing
train_sqrt_diff = train_data['Passengers_Sqrt'].diff().dropna()

# Calculate mean and variance after differencing
sqrt_diff_mean = train_sqrt_diff.mean()
sqrt_diff_std = train_sqrt_diff.std()
sqrt_diff_variance = train_sqrt_diff.var()

print(f"\nDifferenced Data Statistics - Square Root Transformation:")
print(f"Mean: {sqrt_diff_mean:.2f}")
print(f"Standard Deviation: {sqrt_diff_std:.2f}")
print(f"Variance: {sqrt_diff_variance:.2f}")

print(f"\nChange in Statistics - Square Root Transformation:")
print(f"Mean change: {original_mean:.2f} → {sqrt_diff_mean:.2f} (Δ = {abs(original_mean - sqrt_diff_mean):.2f})")
print(f"Variance change: {original_variance:.2f} → {sqrt_diff_variance:.2f} (Δ = {abs(original_variance - sqrt_diff_variance):.2f})")

# ADF test on differenced data
adf_sqrt_diff = adfuller(train_sqrt_diff)
print(f"\nAugmented Dickey-Fuller Test (Differenced Data):")
print(f"ADF Statistic: {adf_sqrt_diff[0]:.4f}")
print(f"p-value: {adf_sqrt_diff[1]:.4f}")

if adf_sqrt_diff[1] <= 0.05:
    print(f"\nData is now STATIONARY (p-value <= 0.05)")
    print(f"Differencing order (d) = 1")
else:
    print(f"\nData may need additional differencing")

# Visualize differenced data
plt.figure(figsize=(12, 6))
plt.plot(train_sqrt_diff.index, train_sqrt_diff, label='Differenced Data', color='purple', linewidth=1.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
plt.title('Differenced Data (First Order) - Square Root Transform', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Differenced Passengers', fontsize=12)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('arima_diff_sqrt.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================================
# STEP 3c: INTEGRATED / DIFFERENCING (d in ARIMA) - Box-Cox applied to original data
# ============================================================================
print("\nSTEP 3c: APPLYING BOX-COX TRANSFORMATION OF DATA PLUS DIFFERENCING TO ACHIEVE STATIONARITY")

# Apply Box-Cox to Data
train_data['Passengers_Boxcox'], lam = boxcox(train_data['Passengers'])
print(f"\nBoxcox Statistic (Lambda): {lam:.2f}")

# Apply first-order differencing
train_boxcox_diff = train_data['Passengers_Boxcox'].diff().dropna()

# Calculate mean and variance after differencing
boxcox_diff_mean = train_boxcox_diff.mean()
boxcox_diff_std = train_boxcox_diff.std()
boxcox_diff_variance = train_boxcox_diff.var()

print(f"\nDifferenced Data Statistics - Box-Cox Transformation:")
print(f"Mean: {boxcox_diff_mean:.2f}")
print(f"Standard Deviation: {boxcox_diff_std:.2f}")
print(f"Variance: {boxcox_diff_variance:.2f}")

print(f"\nChange in Statistics - Box-Cox Transformation:")
print(f"Mean change: {original_mean:.2f} → {boxcox_diff_mean:.2f} (Δ = {abs(original_mean - boxcox_diff_mean):.2f})")
print(f"Variance change: {original_variance:.2f} → {boxcox_diff_variance:.2f} (Δ = {abs(original_variance - boxcox_diff_variance):.2f})")

# ADF test on differenced data
adf_boxcox_diff = adfuller(train_boxcox_diff)
print(f"\nAugmented Dickey-Fuller Test (Differenced Data):")
print(f"ADF Statistic: {adf_boxcox_diff[0]:.4f}")
print(f"p-value: {adf_boxcox_diff[1]:.4f}")

if adf_boxcox_diff[1] <= 0.05:
    print(f"\nData is now STATIONARY (p-value <= 0.05)")
    print(f"Differencing order (d) = 1")
else:
    print(f"\nData may need additional differencing")

# Visualize differenced data
plt.figure(figsize=(12, 6))
plt.plot(train_boxcox_diff.index, train_boxcox_diff, label='Differenced Data', color='purple', linewidth=1.5)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
plt.title('Differenced Data (First Order) - Box-Cox Transform', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Differenced Passengers', fontsize=12)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('arima_diff_box_cox.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# STEP 3d: TRANSFORMATION COMPARISON - SIDE BY SIDE SUMMARY
# ============================================================================
print("\nSTEP 3d: TRANSFORMATION COMPARISON SUMMARY")
print("Comparing the effect of each variance-stabilisation method on the")
print("differenced series, to support the choice of transformation for the ARIMA model.")

fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=False)

# Panel 1: No transformation, first-order differencing
axes[0].plot(train_diff.index, train_diff, color='steelblue', linewidth=1.2)
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
axes[0].set_title(f'No Transformation  |  d=1  |  ADF p={adf_diff[1]:.4f}  '
                  f'|  Var={train_diff.var():.2f}',
                  fontsize=12, fontweight='bold')
axes[0].set_ylabel('Differenced\nPassengers', fontsize=10)

# Panel 2: Log transformation + first-order differencing
axes[1].plot(train_log_diff.index, train_log_diff, color='seagreen', linewidth=1.2)
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
axes[1].set_title(f'Log Transform  |  d=1  |  ADF p={adf_log_diff[1]:.4f}  '
                  f'|  Var={train_log_diff.var():.4f}',
                  fontsize=12, fontweight='bold')
axes[1].set_ylabel('Differenced\nLog(Passengers)', fontsize=10)

# Panel 3: Square-root transformation + first-order differencing
axes[2].plot(train_sqrt_diff.index, train_sqrt_diff, color='darkorange', linewidth=1.2)
axes[2].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
axes[2].set_title(f'Square-Root Transform  |  d=1  |  ADF p={adf_sqrt_diff[1]:.4f}  '
                  f'|  Var={train_sqrt_diff.var():.4f}',
                  fontsize=12, fontweight='bold')
axes[2].set_ylabel('Differenced\n√(Passengers)', fontsize=10)

# Panel 4: Box-Cox transformation + first-order differencing
axes[3].plot(train_boxcox_diff.index, train_boxcox_diff, color='purple', linewidth=1.2)
axes[3].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
axes[3].set_title(f'Box-Cox Transform (λ={lam:.2f})  |  d=1  |  ADF p={adf_boxcox_diff[1]:.4f}  '
                  f'|  Var={train_boxcox_diff.var():.4f}',
                  fontsize=12, fontweight='bold')
axes[3].set_ylabel('Differenced\nBox-Cox(Passengers)', fontsize=10)

for ax in axes:
    ax.set_xlabel('Date', fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle('Variance-Stabilisation Transformation Comparison\n'
             '(All series: First-Order Differenced Training Data)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('arima_transformation_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Print summary table
print(f"\nTransformation Comparison Summary:")
print(f"{'Transformation':<30} {'ADF p-value':>12} {'ADF Result':>16} {'Variance':>12}")
print("-" * 72)
transforms = [
    ("None (raw differencing)",    adf_diff[1],        train_diff.var()),
    ("Log + d=1",                  adf_log_diff[1],    train_log_diff.var()),
    ("Square-Root + d=1",          adf_sqrt_diff[1],   train_sqrt_diff.var()),
    (f"Box-Cox (λ={lam:.2f}) + d=1", adf_boxcox_diff[1], train_boxcox_diff.var()),
]
for name, adf_p, var in transforms:
    result = "Stationary" if adf_p <= 0.05 else "Non-stationary"
    print(f"{name:<30} {adf_p:>12.4f} {result:>16} {var:>12.4f}")


# ============================================================================
# STEP 4: PARTIAL AUTOCORRELATION FUNCTION (PACF) - DETERMINE AR ORDER (p)
# ============================================================================
print("\nSTEP 4: PACF ANALYSIS - DETERMINING AUTOREGRESSION LAGS (p)")

# Calculate PACF
pacf_values = pacf(train_diff, nlags=16)

print(f"\nPACF Values (first 12 lags):")
for i in range(min(13, len(pacf_values))):
    print(f"  Lag {i}: {pacf_values[i]:.4f}")

# Determine significant lags (using 95% confidence interval)
n = len(train_diff)
confidence_interval = 1.96 / np.sqrt(n)
significant_lags_pacf = [i for i in range(1, len(pacf_values)) if abs(pacf_values[i]) > confidence_interval]

print(f"\nConfidence Interval (95%): ±{confidence_interval:.4f}")
print(f"Significant lags in PACF: {significant_lags_pacf[:8]}")  # Show first 8

# Suggested AR order
# suggested_p = len([i for i in range(1, min(6, len(pacf_values))) if abs(pacf_values[i]) > confidence_interval])
suggested_p = 12                # This is user defined based on visual assessment of the PACF plot
print(f"\nSuggested AR order (p): {suggested_p}")
print(f"(Number of significant lags in PACF before cutoff)")

# Visualize PACF
plt.figure(figsize=(12, 6))
plt.stem(range(len(pacf_values)), pacf_values, basefmt=' ')
plt.axhline(y=confidence_interval, color='red', linestyle='--', linewidth=1, label='95% Confidence')
plt.axhline(y=-confidence_interval, color='red', linestyle='--', linewidth=1)
plt.axhline(y=0, color='black', linewidth=0.5)
plt.title('Partial Autocorrelation Function (PACF) - Determines AR(p) Order', fontsize=16, fontweight='bold')
plt.xlabel('Lag', fontsize=12)
plt.ylabel('PACF', fontsize=12)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('arima_pacf.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# STEP 5: AUTOCORRELATION FUNCTION (ACF) - DETERMINE MA ORDER (q)
# ============================================================================
print("\nSTEP 5: ACF ANALYSIS - DETERMINING MOVING AVERAGE LAGS (q)")

# Redefine confidence interval for this block (series may differ from PACF block)
n = len(train_diff)
confidence_interval = 1.96 / np.sqrt(n)

# Calculate ACF
acf_values = acf(train_diff, nlags=16)

print(f"\nACF Values (first 12 lags):")
for i in range(min(13, len(acf_values))):
    print(f"  Lag {i}: {acf_values[i]:.4f}")

# Determine significant lags
significant_lags_acf = [i for i in range(1, len(acf_values)) if abs(acf_values[i]) > confidence_interval]

print(f"\nConfidence Interval (95%): ±{confidence_interval:.4f}")
print(f"Significant lags in ACF: {significant_lags_acf[:8]}")  # Show first 8

# Suggested MA order
# suggested_q = len([i for i in range(1, min(6, len(acf_values))) if abs(acf_values[i]) > confidence_interval])
suggested_q = 12                # This is user defined based on visual assessment of the ACF plot
print(f"\nSuggested MA order (q): {suggested_q}")
print(f"(Number of significant lags in ACF before cutoff)")

# Visualize ACF
plt.figure(figsize=(12, 6))
plt.stem(range(len(acf_values)), acf_values, basefmt=' ')
plt.axhline(y=confidence_interval, color='red', linestyle='--', linewidth=1, label='95% Confidence')
plt.axhline(y=-confidence_interval, color='red', linestyle='--', linewidth=1)
plt.axhline(y=0, color='black', linewidth=0.5)
plt.title('Autocorrelation Function (ACF) - Determines MA(q) Order', fontsize=16, fontweight='bold')
plt.xlabel('Lag', fontsize=12)
plt.ylabel('ACF', fontsize=12)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('arima_acf.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# STEP 5a: ACF / PACF ON BOX-COX TRANSFORMED DATA (CONFIRMATORY)
# ============================================================================
# The exploratory ACF/PACF above used first-order differenced raw data, which is appropriate for understanding the autocorrelation structure of the series.
# Since Box-Cox transformation was identified in Step 3 as producing the best-conditioned stationary series,
# we confirm the ACF/PACF findings using the same series that will enter the optimal ARIMA model.
# For this dataset the 12-month seasonality dominates both series, so the suggested p and q values are expected to be consistent.
# ============================================================================
print("\nSTEP 5b: CONFIRMATORY ACF/PACF ON BOX-COX TRANSFORMED DATA")
print("Verifying that p and q conclusions are consistent with the optimal stationary series using the Box-Cox transformation")

n_bc = len(train_boxcox_diff)
ci_bc = 1.96 / np.sqrt(n_bc)

pacf_bc = pacf(train_boxcox_diff, nlags=16)
acf_bc  = acf(train_boxcox_diff,  nlags=16)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PACF
axes[0].stem(range(len(pacf_bc)), pacf_bc, basefmt=' ')
axes[0].axhline(y= ci_bc, color='red', linestyle='--', linewidth=1, label='95% Confidence')
axes[0].axhline(y=-ci_bc, color='red', linestyle='--', linewidth=1)
axes[0].axhline(y=0, color='black', linewidth=0.5)
axes[0].set_title('PACF — Box-Cox Transformed Data\n(Determines AR order p)',
                  fontsize=13, fontweight='bold')
axes[0].set_xlabel('Lag', fontsize=11)
axes[0].set_ylabel('PACF', fontsize=11)
axes[0].legend(fontsize=10)

# ACF
axes[1].stem(range(len(acf_bc)), acf_bc, basefmt=' ')
axes[1].axhline(y= ci_bc, color='red', linestyle='--', linewidth=1, label='95% Confidence')
axes[1].axhline(y=-ci_bc, color='red', linestyle='--', linewidth=1)
axes[1].axhline(y=0, color='black', linewidth=0.5)
axes[1].set_title('ACF — Box-Cox Transformed Data\n(Determines MA order q)',
                  fontsize=13, fontweight='bold')
axes[1].set_xlabel('Lag', fontsize=11)
axes[1].set_ylabel('ACF', fontsize=11)
axes[1].legend(fontsize=10)

plt.suptitle('Confirmatory ACF/PACF — Box-Cox Transformed, First-Order Differenced Data',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('arima_acf_pacf_boxcox.png', dpi=150, bbox_inches='tight')
plt.show()

# Report whether conclusions are consistent
sig_p_bc = [i for i in range(1, len(pacf_bc)) if abs(pacf_bc[i]) > ci_bc]
sig_q_bc = [i for i in range(1, len(acf_bc))  if abs(acf_bc[i])  > ci_bc]

print(f"\nBox-Cox PACF — significant lags: {sig_p_bc[:8]}")
print(f"Box-Cox ACF  — significant lags: {sig_q_bc[:8]}")
print(f"\nOriginal PACF — significant lags: {significant_lags_pacf[:8]}")
print(f"Original ACF  — significant lags: {significant_lags_acf[:8]}")
print(f"\nConclusion: p and q suggestions are "
      f"{'CONSISTENT' if sig_p_bc[:1] == significant_lags_pacf[:1] else 'DIFFERENT'} "
      f"between raw and Box-Cox transformed series.")

# ============================================================================
# STEP 6: BUILD AND FIT ARIMA MODEL
# ============================================================================
print("\nSTEP 6: BUILDING ARIMA MODEL")

# ARIMA parameters: (p, d, q)
# p = AR order from PACF
# d = differencing order
# q = MA order from ACF
# p, d, q = suggested_p, 1, suggested_q

# Common good ARIMA parameters are (1,1,1) or (2,1,2)
p, d, q = 12, 2, 12             # User defined

print(f"\nARIMA Parameters:")
print(f"p (AR order): {p} - Autoregression lags")
print(f"d (Differencing): {d} - Order of differencing")
print(f"q (MA order): {q} - Moving average lags")
# print(f"\nFitting ARIMA({p},{d},{q}) model:")

# Fit ARIMA model
# Uncomment one of the following 4 lines to determine which transformation is applied, if at all
# model = ARIMA(train_data['Passengers'], order=(p, d, q))            # To fit un transformed data
model = ARIMA(train_data['Passengers_Boxcox'], order=(p, d, q))     # To fit data with Box-Cox transformation
# model = ARIMA(train_data['Passengers_Sqrt'], order=(p, d, q))       # To fit data with Square-root transformation
# model = ARIMA(train_data['Passengers_Log'], order=(p, d, q))        # To fit data with Log transformation
fitted_model = model.fit()

print(f"\nARIMA({p},{d},{q}) model Summary:")
print(fitted_model.summary())

# ============================================================================
# STEP 7: MAKE PREDICTIONS WITH CONFIDENCE INTERVALS
# ============================================================================
print("\nSTEP 7: GENERATING FORECASTS WITH CONFIDENCE INTERVALS")

forecast_steps = len(test_data)

# Use get_forecast to obtain prediction intervals alongside point forecasts
forecast_result = fitted_model.get_forecast(steps=forecast_steps)
forecast = forecast_result.predicted_mean
forecast_ci = forecast_result.conf_int(alpha=0.05)   # 95% prediction intervals

# Uncomment the appropriate inverse-transform lines if a transformation was applied in Step 6:
# --- Box-Cox inverse:
forecast = inv_boxcox(forecast, lam)
forecast_ci = forecast_ci.apply(lambda col: inv_boxcox(col, lam))
# --- Square-root inverse:
# forecast = forecast ** 2
# forecast_ci = forecast_ci ** 2
# --- Log inverse:
# forecast = np.exp(forecast)
# forecast_ci = np.exp(forecast_ci)

print(f"\nGenerated {forecast_steps} forecast values with 95% prediction intervals")
print(f"\nFirst 5 predictions:")
for i in range(min(5, len(forecast))):
    lower = forecast_ci.iloc[i, 0]
    upper = forecast_ci.iloc[i, 1]
    print(f"  {test_data.index[i].strftime('%Y-%m')}: {forecast.iloc[i]:.2f}  "
          f"[95% CI: {lower:.2f} – {upper:.2f}]")

# ============================================================================
# STEP 8: EVALUATE MODEL PERFORMANCE
# ============================================================================
print("\nSTEP 8: MODEL EVALUATION METRICS")

print(f"\nResults for model: ARIMA({p},{d},{q})")

# Calculate metrics
mse = mean_squared_error(test_data['Passengers'], forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_data['Passengers'], forecast)
r2 = r2_score(test_data['Passengers'], forecast)

# Calculate MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((test_data['Passengers'] - forecast) / test_data['Passengers'])) * 100

print(f"\nPerformance Metrics:")
print(f"  Mean Squared Error (MSE): {mse:.2f}")
print(f"  Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"  Mean Absolute Error (MAE): {mae:.2f}")
print(f"  R² Score: {r2:.4f}")
print(f"  Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

print(f"\nInterpretation:")
print(f"  RMSE of {rmse:.2f} means predictions are off by ~{rmse:.0f} passengers on average")
print(f"  R² of {r2:.4f} indicates the model explains {r2*100:.1f}% of variance")
print(f"  MAPE of {mape:.2f}% shows average prediction error percentage")

# ============================================================================
# STEP 8a: IN-SAMPLE RESIDUAL DIAGNOSTICS (MODEL VALIDATION)
# ============================================================================
print("\nSTEP 8a: IN-SAMPLE RESIDUAL DIAGNOSTICS")
print("These diagnostics check whether the model has adequately captured the")
print("structure in the training data. Well-specified residuals should be:")
print("  - Uncorrelated (ACF shows no significant lags)")
print("  - Approximately normally distributed")
print("  - Centred on zero with constant variance")

# Statsmodels built-in four-panel diagnostic plot
fig = fitted_model.plot_diagnostics(figsize=(14, 10))
fig.suptitle(f'ARIMA({p},{d},{q}) - In-Sample Residual Diagnostics',
             fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('arima_model_diagnostics.png', dpi=150, bbox_inches='tight')
plt.show()

# Extract residuals for summary statistics
model_residuals = fitted_model.resid
print(f"\nIn-Sample Residual Statistics:")
print(f"  Mean:    {model_residuals.mean():.4f}  (should be close to 0)")
print(f"  Std Dev: {model_residuals.std():.4f}")
print(f"  Skewness: {model_residuals.skew():.4f}")
print(f"  Kurtosis: {model_residuals.kurtosis():.4f}")

# Ljung-Box test for residual autocorrelation
lb_result = acorr_ljungbox(model_residuals.dropna(), lags=[10, 20], return_df=True)
print(f"\nLjung-Box Test (tests for residual autocorrelation):")
print(f"  Null hypothesis: residuals are uncorrelated (white noise)")
print(lb_result.to_string())
if lb_result['lb_pvalue'].iloc[-1] > 0.05:
    print(f"\n  Result: Cannot reject H₀ → Residuals appear to be white noise ✓")
else:
    print(f"\n  Result: Reject H₀ → Residuals show remaining autocorrelation ✗")
    print(f"  Consider adjusting p or q parameters.")

# ============================================================================
# STEP 8b: ACF OF IN-SAMPLE RESIDUALS
# ============================================================================
print("\nSTEP 8b: ACF OF IN-SAMPLE RESIDUALS")
print("A well-specified model should produce residuals that are white noise.")
print("No ACF lags should exceed the confidence interval.")

acf_resid = acf(model_residuals.dropna(), nlags=20)
n_resid = len(model_residuals.dropna())
ci_resid = 1.96 / np.sqrt(n_resid)

plt.figure(figsize=(12, 6))
plt.stem(range(len(acf_resid)), acf_resid, basefmt=' ')
plt.axhline(y=ci_resid,  color='red', linestyle='--', linewidth=1, label='95% Confidence Interval')
plt.axhline(y=-ci_resid, color='red', linestyle='--', linewidth=1)
plt.axhline(y=0, color='black', linewidth=0.5)
plt.title(f'ACF of In-Sample Residuals - ARIMA({p},{d},{q})', fontsize=16, fontweight='bold')
plt.xlabel('Lag', fontsize=12)
plt.ylabel('ACF', fontsize=12)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('arima_residuals_acf.png', dpi=150, bbox_inches='tight')
plt.show()

# Interpret the result
significant_resid_lags = [i for i in range(1, len(acf_resid)) if abs(acf_resid[i]) > ci_resid]
if significant_resid_lags:
    print(f"\nSignificant lags found: {significant_resid_lags}")
    print(f"Remaining autocorrelation detected — consider adjusting p or q.")
else:
    print(f"\nNo significant lags found — residuals are consistent with white noise.")
    print(f"The model has adequately captured the autocorrelation structure.")

# ============================================================================
# STEP 9: VISUALISE PREDICTIONS VS ACTUAL
# ============================================================================
print("\nSTEP 9: VISUALISING RESULTS")

# Plot predictions vs actual
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data['Passengers'], label='Training Data', linewidth=2, color='blue')
plt.plot(test_data.index, test_data['Passengers'], label='Actual Test Data', linewidth=2, color='green')
plt.plot(test_data.index, forecast, label='ARIMA Forecast', linewidth=2, color='red', linestyle='--')
plt.fill_between(test_data.index,
                 forecast_ci.iloc[:, 0],
                 forecast_ci.iloc[:, 1],
                 color='red', alpha=0.15, label='95% Prediction Interval')
plt.title(f'ARIMA({p},{d},{q}) Model - Forecast vs Actual', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Passengers', fontsize=12)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('arima_pred_1.png', dpi=150, bbox_inches='tight')
plt.show()

# Residual plot
# residuals = test_data['Passengers'].values - forecast.values
forecast_aligned = forecast.reindex(test_data.index)
residuals = (test_data['Passengers'] - forecast_aligned).values
plt.figure(figsize=(12, 6))
plt.scatter(range(len(residuals)), residuals, alpha=0.7, color='purple', s=50)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.title('Prediction Residuals (Actual - Predicted)', fontsize=16, fontweight='bold')
plt.xlabel('Observation', fontsize=12)
plt.ylabel('Residual', fontsize=12)
plt.tight_layout()
plt.savefig('arima_residual_1.png', dpi=150, bbox_inches='tight')
plt.show()

# Distribution of residuals
plt.figure(figsize=(12, 6))
sns.histplot(residuals, kde=True, bins=15, color='teal')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
plt.title('Distribution of Prediction Residuals', fontsize=16, fontweight='bold')
plt.xlabel('Residual Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('arima_residual_histo_1.png', dpi=150, bbox_inches='tight')
plt.show()

# # ============================================================================
# # STEP 10: AUTOMATED PARAMETER SELECTION (AUTO-ARIMA)
# # ============================================================================
# # This section uses pmdarima's auto_arima to automatically identify ARIMA parameters.
# # It is provided as a reference point to compare against the manually derived parameters above.
# # The exploratory analysis in Steps 1-9 remains the primary methodology.
# # ============================================================================
# print("\nSTEP 10: AUTOMATED ARIMA PARAMETER SELECTION (auto_arima)")
# print("Note: This step is for reference — it shows what an automated search recommends")
# print("      versus the parameters derived through the exploratory analysis above.")
#
# try:
#     import pmdarima as pm
#
#     print("\nRunning auto_arima on training data")
#     print("Searching over p ∈ [0,3], d ∈ [0,2], q ∈ [0,3] with seasonal=False")
#
#     auto_model = pm.auto_arima(
#         train_data['Passengers'],
#         start_p=0, max_p=3,
#         start_q=0, max_q=3,
#         d=None,                   # Let auto_arima determine d via ADF test
#         max_d=2,
#         seasonal=False,           # Non-seasonal ARIMA only (SARIMA is a next-step recommendation)
#         information_criterion='aic',
#         stepwise=True,
#         suppress_warnings=True,
#         error_action='ignore',
#         trace=True                # Shows the search progress
#     )
#
#     auto_p, auto_d, auto_q = auto_model.order
#     print(f"\nauto_arima recommended order: ARIMA({auto_p},{auto_d},{auto_q})")
#     print(f"AIC: {auto_model.aic():.2f}")
#
#     # Evaluate auto_arima model on test set
#     auto_forecast = auto_model.predict(n_periods=len(test_data))
#     auto_mse  = mean_squared_error(test_data['Passengers'], auto_forecast)
#     auto_rmse = np.sqrt(auto_mse)
#     auto_mae  = mean_absolute_error(test_data['Passengers'], auto_forecast)
#     auto_r2   = r2_score(test_data['Passengers'], auto_forecast)
#     auto_mape = np.mean(np.abs((test_data['Passengers'] - auto_forecast)
#                                / test_data['Passengers'])) * 100
#
#     print(f"\nauto_arima ARIMA({auto_p},{auto_d},{auto_q}) Test Performance:")
#     print(f"  RMSE:  {auto_rmse:.2f}")
#     print(f"  MAE:   {auto_mae:.2f}")
#     print(f"  R²:    {auto_r2:.4f}")
#     print(f"  MAPE:  {auto_mape:.2f}%")
#
#     # Compare against manually tuned model
#     print(f"\nComparison: Manual vs Auto-ARIMA")
#     print(f"{'Model':<30} {'Order':>12} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'MAPE':>8}")
#     print("-" * 76)
#     print(f"{'Manual (best)':<30} {'(' + str(p) + ',' + str(d) + ',' + str(q) + ')':>12}"
#           f" {rmse:>8.2f} {mae:>8.2f} {r2:>8.4f} {mape:>7.2f}%")
#     print(f"{'auto_arima':<30} {'(' + str(auto_p) + ',' + str(auto_d) + ',' + str(auto_q) + ')':>12}"
#           f" {auto_rmse:>8.2f} {auto_mae:>8.2f} {auto_r2:>8.4f} {auto_mape:>7.2f}%")
#
#     # Plot auto_arima forecast
#     plt.figure(figsize=(12, 6))
#     plt.plot(train_data.index, train_data['Passengers'],
#              label='Training Data', linewidth=2, color='blue')
#     plt.plot(test_data.index, test_data['Passengers'],
#              label='Actual Test Data', linewidth=2, color='green')
#     plt.plot(test_data.index, auto_forecast,
#              label=f'auto_arima ARIMA({auto_p},{auto_d},{auto_q})',
#              linewidth=2, color='darkorange', linestyle='--')
#     plt.title(f'auto_arima Forecast: ARIMA({auto_p},{auto_d},{auto_q})',
#               fontsize=16, fontweight='bold')
#     plt.xlabel('Date', fontsize=12)
#     plt.ylabel('Number of Passengers', fontsize=12)
#     plt.legend(fontsize=11)
#     plt.tight_layout()
#     plt.savefig('arima_auto_forecast.png', dpi=150, bbox_inches='tight')
#     plt.show()
#
#     print(f"\nNote: auto_arima searches a constrained space and uses AIC as the criterion.")
#     print(f"      The manually tuned model may outperform it by explicitly leveraging")
#     print(f"      domain knowledge (12-month seasonality) and variance stabilisation.")
#
# except ImportError:
#     print("\nAlternatively, a manual grid search can be implemented using:")
#     print("  from itertools import product")
#     print("  for p, d, q in product(range(0,3), range(1,3), range(0,3)):")
#     print("      model = ARIMA(train_data['Passengers'], order=(p,d,q)).fit()")
#     print("      print(p, d, q, model.aic())")


# Track time to complete process
t1 = time.time()  # Add at end of process
timetaken1 = t1 - t0
print(f"\nTime Taken: {timetaken1:.4f} seconds")
