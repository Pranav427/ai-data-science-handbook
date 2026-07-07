# 📘 Time Series Analysis: Temporal Modeling & Forecasting

An engineering reference for stationarity tests, autoregressive models (ARIMA), seasonal decomposition, and temporal validation splits.

---

## 💡 Engineering Intuition & Stationarity

Time series data contains sequence dependencies where chronological order must be preserved.

- **Stationarity**: A time series is stationary if its statistical properties (mean, variance, autocorrelation) do not change over time. Most forecasting models (like ARIMA) require the series to be stationary before training.
- **Augmented Dickey-Fuller (ADF) Test**: A statistical test used to check for stationarity. If the \(p\)-value is \(< 0.05\), we reject the null hypothesis and assume the series is stationary.
- **Differencing (\(d\) parameter)**: Subtracting the current value from the previous value (\(X_t - X_{t-1}\)) to remove trend or seasonality, making the series stationary.
- **ARIMA (\(p, d, q\))**:
  - **\(p\)**: Autoregressive order (lags of the target).
  - **\(d\)**: Differencing order.
  - **\(q\)**: Moving average order (lags of the forecast errors).

---

## 🛠 Best-Practice Stationarity & Forecasting Example

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# Simulate a non-stationary time series (random walk with trend)
np.random.seed(42)
steps = np.random.normal(loc=0.1, scale=1.0, size=100)
time_series = np.cumsum(steps)

# 1. Run Augmented Dickey-Fuller Test
result = adfuller(time_series)
print(f"ADF Statistic: {result[0]:.4f} | p-value: {result[1]:.4f}")
# If p-value > 0.05, series is non-stationary

# 2. Fit ARIMA(1, 1, 1) Model
model = ARIMA(time_series, order=(1, 1, 1))
fitted_model = model.fit()

# Forecast next 3 time steps
forecast = fitted_model.forecast(steps=3)
print("Forecasted steps:", forecast)
```

---

## ⚠️ Common Pitfalls & Debugging Tips

### 1. Data Leakage in Validation
*   **The Bug**: Using standard random K-Fold cross-validation on time-series data. This uses future data points to train the model to predict past data points, yielding artificially high accuracy scores.
*   **The Fix**: Use sliding-window or expanding-window validation (like Scikit-Learn's `TimeSeriesSplit`) to ensure training data always precedes validation data chronologically.

### 2. Spurious Regression on Trends
*   **The Bug**: Regressing two completely unrelated trending variables (e.g. ice cream sales and sunscreen sales over a year) and getting a high \(R^2\) because they share a seasonal trend.
*   **The Fix**: Decompose the series into Trend, Seasonality, and Residual components first, or difference the data to analyze variance correlation rather than absolute values.

---

## ⚡ Real-World & Project Connections

*   **PathPilot**: Foreknows temporal traffic densities by evaluating history. Temporal metrics are modeled to anticipate traffic slowdowns during morning/evening rush hours.

---

## 🔗 Further Reading & Reference Links

- [Forecasting: Principles and Practice (Hyndman & Athanasopoulos)](https://otexts.com/fpp3/)
- [Statsmodels Time Series Module Reference](https://www.statsmodels.org/stable/tsa.html)
