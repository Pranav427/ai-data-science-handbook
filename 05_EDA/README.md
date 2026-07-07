# 📘 Exploratory Data Analysis (EDA): Data Auditing & Profiling

A guide to exploratory data analysis, systematic data auditing, and finding distribution shifts in production machine learning pipelines.

---

## 💡 Engineering Intuition & Data Auditing

Before training models, you must audit the data. EDA is not just about drawing plots; it is about finding structural failures, bias, and target leakage.

- **Target Leakage**: The presence of features in the training data that contain information about the target variable which would not be available during real-time inference. (e.g., including "discharge_date" when predicting "hospital_readmission").
- **Missing Data Mechanisms**:
  - **MCAR** (Missing Completely at Random): The missingness is completely random. Imputation is safe.
  - **MAR** (Missing at Random): Missingness depends on other observed variables. Grouped imputation (imputing based on other features) works best.
  - **MNAR** (Missing Not at Random): The value is missing because of the value itself (e.g., high-income earners hiding salary). This requires adding a missingness indicator column (`is_missing`) to preserve the pattern for the model.
- **Correlation & Collinearity**: High correlation between independent variables (collinearity) destabilizes regression models and distorts feature importances. Use Spearman rank correlation for non-linear relationships.

---

## 🛠 Best-Practice Data Audit Snippet

```python
import pandas as pd
import numpy as np

def audit_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs a high-level programmatic data audit for missingness and cardinality.
    """
    audit_report = pd.DataFrame({
        "data_type": df.dtypes,
        "null_count": df.isnull().sum(),
        "null_percent": (df.isnull().sum() / len(df)) * 100,
        "unique_values": df.nunique(),
    })
    return audit_report

# Example Usage
df = pd.DataFrame({
    "age": [25, np.nan, 45, 30, 22],
    "city": ["NY", "SF", "NY", "LA", np.nan],
    "target": [1, 0, 1, 0, 0]
})
print(audit_dataset(df))
```

---

## ⚠️ Common Pitfalls & Debugging Tips

### 1. Data Leakage via Preprocessing
*   **The Bug**: Calculating the imputation mean or scaling parameters over the *entire* dataset before performing train/test split. Information from the test set leaks into the training step, leading to overoptimistic validation scores.
*   **The Fix**: Always split train/test first, fit the imputer/scaler on the **training set only**, and then transform both:
    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.impute import SimpleImputer

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    imputer = SimpleImputer(strategy="median")
    X_train_clean = imputer.fit_transform(X_train)
    X_test_clean = imputer.transform(X_test)  # Transform only
    ```

### 2. Plotting Millions of Rows
*   **The Bug**: Running `sns.pairplot(df)` on a DataFrame with hundreds of thousands of rows. This freezes the Jupyter kernel or browser tab due to rendering thousands of SVG nodes.
*   **The Fix**: Downsample the data (`df.sample(n=1000)`) specifically for visualization purposes, or use aggregate hexbin plots.

---

## ⚡ Real-World & Project Connections

*   **Medical Condition Classification**: EDA helps spot target leakage where billing codes or diagnostic identifiers (written after the patient's visit) are accidentally included as inputs to predict the patient's primary disease classification.

---

## 🔗 Further Reading & Reference Links

- [Seaborn Visualization Guide](https://seaborn.pydata.org/)
- [Preventing Data Leakage in ML](https://scikit-learn.org/stable/common_obstacles.html#data-leakage)
