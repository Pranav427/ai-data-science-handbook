# 📘 Basic Statistics – I: Descriptive Statistics & Data Distributions

A quick reference for descriptive statistics, data distributions, and their engineering implications in machine learning pipelines.

---

## 💡 Engineering Intuition & Distributions

Data is rarely clean. Understanding descriptive statistics is the first step in validating whether a dataset is representative of the real world.

- **Robust Statistics**: 
  - **Mean** is highly sensitive to outliers. 
  - **Median** is robust against extreme values. 
  - In skewed distributions (e.g., medical costs or user activity logs), always rely on the median and IQR (Interquartile Range) rather than the mean and standard deviation.
- **Central Limit Theorem (CLT)**: The CLT states that the sum (or average) of a large number of independent random variables will approximate a normal (Gaussian) distribution, regardless of the underlying distribution. This justifies why we can safely assume Gaussian noise in regression objectives (like Mean Squared Error).
- **Standard Deviation vs. Variance**: Variance measures the spread of the data, while standard deviation brings the units back to the original scale of the features, making it directly interpretable.

---

## 🛠 Best-Practice Implementation Example

```python
import numpy as np
from scipy import stats

# Simulating a highly skewed distribution (outliers present)
data = np.concatenate([np.random.normal(50, 5, 950), np.random.normal(500, 50, 50)])

# Compute standard and robust statistics
mean = np.mean(data)
median = np.median(data)

# Robust dispersion: Median Absolute Deviation (MAD)
mad = stats.median_abs_deviation(data)

# Interquartile Range (IQR)
q75, q25 = np.percentile(data, [75 ,25])
iqr = q75 - q25

print(f"Mean: {mean:.2f} (skewed by outliers)")
print(f"Median: {median:.2f} (robust baseline)")
print(f"MAD: {mad:.2f} | IQR: {iqr:.2f}")
```

---

## ⚠️ Common Pitfalls & Debugging Tips

### 1. Treating Skewed Distributions as Normal
*   **The Bug**: Applying the empirical rule (\(3\sigma\) rule) to detect outliers in log-normal or skewed data. This will flag valid tail-end points as anomalies.
*   **The Fix**: Transform the distribution (e.g., using a log-transform or Box-Cox transformation) before applying standard-deviation-based outlier detection, or use IQR-based detection.

### 2. Bessel's Correction Confusion (\(N\) vs. \(N-1\))
*   **The Bug**: Underestimating variance by dividing by the sample size \(N\) instead of the degrees of freedom \(N-1\) for sample variance calculations.
*   **The Fix**: Use `ddof=1` in NumPy/Pandas variance and standard deviation calls:
    ```python
    sample_var = np.var(data, ddof=1)  # Correct for sample estimations
    ```

---

## ⚡ Real-World & Project Connections

*   **Telemetry Logs**: When measuring server response latencies for APIs (like **PathPilot**'s routing endpoints), report the **P95** or **P99** latencies. The mean latency is misleading due to occasional spikes.

---

## 🔗 Further Reading & Reference Links

- [SciPy Descriptive Statistics](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Bessel's Correction Details](https://en.wikipedia.org/wiki/Bessel%27s_correction)
