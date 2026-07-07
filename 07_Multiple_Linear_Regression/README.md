# 📘 Multiple Linear Regression: Continuous Target Prediction & Diagnosis

An engineering reference for multiple linear regression, OLS estimation, assumptions verification, and model evaluation.

---

## 💡 Engineering Intuition & OLS

Linear Regression is the simplest baseline for predicting a continuous variable. It models a linear relationship between features and the target.

- **Ordinary Least Squares (OLS)**: OLS estimates parameters by minimizing the sum of squared residuals:
  \[
  \hat{\beta} = (X^T X)^{-1} X^T y
  \]
- **Multicollinearity & VIF**: When input features are highly correlated, the matrix \(X^T X\) becomes near-singular, making the estimated coefficients (\(\beta\)) highly unstable and uninterpretable. Use the **Variance Inflation Factor (VIF)** to detect and drop collinear features (VIF > 5 or 10 indicating high collinearity).
- **\(R^2\) vs. Adjusted \(R^2\)**:
  - **\(R^2\)** increases or stays the same as you add more features, even if they are noise.
  - **Adjusted \(R^2\)** penalizes the score for adding non-predictive features, providing a truer estimate of model generalizability.

---

## 🛠 Best-Practice Regression & VIF Example

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# 1. Fit OLS with Statsmodels for Detailed Diagnostics
X_raw = pd.DataFrame({
    "feature_1": np.random.rand(100),
    "feature_2": np.random.rand(100),
    "collinear_feature": np.random.rand(100)
})
# Make collinear_feature almost identical to feature_1
X_raw["collinear_feature"] = X_raw["feature_1"] * 2 + np.random.normal(0, 0.01, 100)
y = X_raw["feature_1"] * 3 + np.random.rand(100)

X_with_const = sm.add_constant(X_raw)
model = sm.OLS(y, X_with_const).fit()

# 2. Check Multicollinearity (VIF)
vifs = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
print("VIFs (index 0 is constant):", np.round(vifs, 2))  # Will show high values for feature_1 & collinear
```

---

## ⚠️ Common Pitfalls & Debugging Tips

### 1. Extrapolation Failure
*   **The Bug**: Applying a linear model to make predictions far outside the range of the training feature values (e.g., predicting real estate prices for a 100-room mansion using a dataset of 1-4 bedroom homes).
*   **The Fix**: Constrain model inputs using domain validation bounds, or use tree-based models that clip predictions naturally at their leaf levels.

### 2. High VIF Distorting Feature Importance
*   **The Bug**: Recommending feature importance or business actions based on unstable, high-VIF coefficient signs (e.g., a positive impact showing up as negative).
*   **The Fix**: Drop the highly correlated variables, combine them using PCA, or apply regularization (L1/L2 ridge or lasso regression) to penalize coefficient magnitudes.

---

## ⚡ Real-World & Project Connections

*   **Baseline Latency Forecasting**: Used in routing software (like **PathPilot**) as a fast, low-compute baseline estimator to predict travel duration before calling more expensive spatial algorithms.

---

## 🔗 Further Reading & Reference Links

- [Scikit-Learn Linear Regression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Statsmodels Regression Diagnostics](https://www.statsmodels.org/stable/regression.html)
