# 📘 Data Transformation: Preprocessing & Scaling Pipelines

An engineering reference for data normalization, standardization, categorical encoding, and feature engineering pipelines.

---

## 💡 Engineering Intuition & Data Scaling

Raw values feed directly into model calculations. Scaling and encoding choices dictate model stability, accuracy, and latency.

- **Normalization (Min-Max Scaling)**: Scales values to a fixed range (usually \([0, 1]\)). Essential for models sensitive to absolute values (e.g., Image processing, Neural Networks, or K-Nearest Neighbors).
  \[
  x_{\text{scaled}} = \frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}}
  \]
- **Standardization (Z-score Scaling)**: Centers data around a mean of 0 with a standard deviation of 1. Recommended for algorithms that assume normally distributed data or rely on gradient descent (e.g., Support Vector Machines, Linear/Logistic Regression, Neural Networks).
  \[
  z = \frac{x - \mu}{\sigma}
  \]
- **One-Hot Encoding vs. Cardinality**: One-hot encoding creates a binary column for every category. For high-cardinality features (e.g., "ZIP_code" with 10,000 values), this leads to cardinality explosion, increasing training time and memory usage. Use Target Encoding or hashing trick instead.

---

## 🛠 Best-Practice Preprocessing Pipeline Example

```python
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

# Raw input data
df = pd.DataFrame({
    "patient_age": [45, 34, 82, 29],
    "cholesterol": [220, 180, 290, 195],
    "gender": ["F", "M", "F", "M"],
    "blood_group": ["A+", "O-", "B+", "O+"]
})

# Define transformers for numerical and categorical groups
num_features = ["patient_age", "cholesterol"]
cat_features = ["gender", "blood_group"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
    ]
)

# Fit pipeline (O(N) computation)
processed_matrix = preprocessor.fit_transform(df)
print("Processed Matrix Shape:", processed_matrix.shape)
```

---

## ⚠️ Common Pitfalls & Debugging Tips

### 1. Hardcoding Categorical Categories
*   **The Bug**: A categorical column in production receives a category that wasn't present during training, throwing an `ValueError: Found unknown categories...` error.
*   **The Fix**: Use `OneHotEncoder(handle_unknown="ignore")` to ignore unknown categories by returning all zeros for that feature.

### 2. Leaking Stats in Pipeline Transforms
*   **The Bug**: Applying scaling to the entire dataset before splitting train/test.
*   **The Fix**: Integrate scaling directly into a Pipeline to guarantee `fit()` runs only on training folds:
    ```python
    # Pipeline handles splitting boundaries correctly
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    ```

---

## ⚡ Real-World & Project Connections

*   **Medical Condition Classification**: Structured clinical values (e.g. heart rate, laboratory diagnostics) are scaled using standardization to avoid dominant weight updates from large-scale measurements.

---

## 🔗 Further Reading & Reference Links

- [Scikit-Learn Preprocessing Guide](https://scikit-learn.org/stable/modules/preprocessing.html)
- [How to Handle Categorical Features](https://towardsdatascience.com/one-hot-encoding-is-making-your-database-cry-b684dbed7fb5)
