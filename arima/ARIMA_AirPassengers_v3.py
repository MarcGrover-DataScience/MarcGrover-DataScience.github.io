# ARIMA Time Series Forecasting - Proof of Concept
# https://www.kaggle.com/code/sunaysawant/air-passengers-time-series-arima/notebook
# Dataset: Air Passengers

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
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
plt.show()

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
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(rolling_var.index, rolling_var, label='Rolling Variance (12 months)', linewidth=2, color='green')
plt.title('Original Data - Rolling Variance (Non-Stationary)', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Variance', fontsize=12)
plt.legend(fontsize=11)
plt.tight_layout()
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
plt.show()


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
plt.show()


# ============================================================================
# STEP 3c: INTEGRATED / DIFFERENCING (d in ARIMA) - Box-Cox applied to original data
# ============================================================================
print("\nSTEP 3b: APPLYING BOX-COX TRANSFORMATION OF DATA PLUS DIFFERENCING TO ACHIEVE STATIONARITY")

# Apply Box-Cox to Data
train_data['Passengers_Boxcox'], lam = boxcox(train_data['Passengers'])
print(f"\nBoxcox Statistic (Lambda): {lam:.2f}")

# # Take Logarithm of Original Data
# train_data['Passengers_Sqrt'] = np.sqrt(train_data['Passengers'])

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
plt.show()


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
plt.show()

# ============================================================================
# STEP 5: AUTOCORRELATION FUNCTION (ACF) - DETERMINE MA ORDER (q)
# ============================================================================
print("\nSTEP 5: ACF ANALYSIS - DETERMINING MOVING AVERAGE LAGS (q)")

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
plt.show()

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
p, d, q = 12, 1, 12             # User defined

print(f"\nARIMA Parameters:")
print(f"p (AR order): {p} - Autoregression lags")
print(f"d (Differencing): {d} - Order of differencing")
print(f"q (MA order): {q} - Moving average lags")
# print(f"\nFitting ARIMA({p},{d},{q}) model:")

# Fit ARIMA model
# Uncomment one of the following 4 lines to determine which transformation is applied, if at all
model = ARIMA(train_data['Passengers'], order=(p, d, q))            # To fit un transformed data
# model = ARIMA(train_data['Passengers_Boxcox'], order=(p, d, q))     # To fit data with Box-Cox transformation
# model = ARIMA(train_data['Passengers_Sqrt'], order=(p, d, q))       # To fit data with Square-root transformation
# model = ARIMA(train_data['Passengers_Log'], order=(p, d, q))        # To fit data with Log transformation
fitted_model = model.fit()

print(f"\nARIMA({p},{d},{q}) model Summary:")
print(fitted_model.summary())

# ============================================================================
# STEP 7: MAKE PREDICTIONS
# ============================================================================
print("\nSTEP 7: GENERATING FORECASTS")

# Forecast for the test period
forecast_steps = len(test_data)
forecast = fitted_model.forecast(steps=forecast_steps)
# Uncomment one of the following 3 lines if transformation applied to data in Step 6 above
# forecast = inv_boxcox(forecast,lam)                   # To inverse the forecasts using boxcox
# forecast = forecast**2                                # To inverse the forecasts using square-root
# forecast = np.exp(forecast)                           # To inverse the forecasts using Log

print(f"\nGenerated {forecast_steps} forecast values")
print(f"\nFirst 5 predictions:")
for i in range(min(5, len(forecast))):
    print(f"  {test_data.index[i].strftime('%Y-%m')}: {forecast.iloc[i]:.2f}")

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
# STEP 9: VISUALISE PREDICTIONS VS ACTUAL
# ============================================================================
print("\nSTEP 9: VISUALISING RESULTS")

# Plot predictions vs actual
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data['Passengers'], label='Training Data', linewidth=2, color='blue')
plt.plot(test_data.index, test_data['Passengers'], label='Actual Test Data', linewidth=2, color='green')
plt.plot(test_data.index, forecast, label='ARIMA Forecast', linewidth=2, color='red', linestyle='--')
plt.title(f'ARIMA({p},{d},{q}) Model - Forecast vs Actual', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Passengers', fontsize=12)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

# Residual plot
residuals = test_data['Passengers'].values - forecast.values
plt.figure(figsize=(12, 6))
plt.scatter(range(len(residuals)), residuals, alpha=0.7, color='purple', s=50)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.title('Prediction Residuals (Actual - Predicted)', fontsize=16, fontweight='bold')
plt.xlabel('Observation', fontsize=12)
plt.ylabel('Residual', fontsize=12)
plt.tight_layout()
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
plt.show()

# Track time to complete process
t1 = time.time()  # Add at end of process
timetaken1 = t1 - t0
print(f"\nTime Taken: {timetaken1:.4f} seconds")
#
# print("\nARIMA ANALYSIS COMPLETE")
# print(f"\nFinal Model: ARIMA({p},{d},{q})")
# print(f"RMSE: {rmse:.2f} passengers")
# print(f"R² Score: {r2:.4f}")
