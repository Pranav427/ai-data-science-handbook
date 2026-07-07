# 📘 Data Structures, NumPy & Pandas: High-Performance Data Wrangling

An engineering reference for memory-efficient data structures, vectorized array operations, and structured data manipulation.

---

## 💡 Engineering Intuition & Trade-offs

AI models consume enormous amounts of data. Processing data using standard Python loops creates high CPU overhead. Vectorization is essential for production workloads.

- **Vectorization vs. Iteration**: NumPy and Pandas delegate element-wise operations to precompiled C code, transforming \(O(N)\) loops into parallelized SIMD hardware instructions. **Rule of thumb: Never use `for` loops to modify rows in Pandas.**
- **NumPy Broadcasting**: When operating on arrays of different shapes, NumPy automatically stretches the smaller array to match the larger array's dimensions without copying data in memory.
- **Pandas Memory Management**: Pandas defaults to loading numerical data into high-overhead types (`float64`, `int64`). Downcasting to `float32`, `int8`, or casting strings to the `category` data type can reduce DataFrame memory usage by up to **80%**.

---

## 🛠 Best-Practice Implementation Example

```python
import numpy as np
import pandas as pd

# 1. Efficient NumPy array operations (Vectorized)
X = np.random.rand(1000, 10)
weights = np.random.rand(10)
# Fast dot product (O(N * D))
predictions = np.dot(X, weights)

# 2. Memory-Optimized Pandas DataFrame
data = {
    "patient_id": [f"P_{i}" for i in range(1000)],
    "age": np.random.randint(18, 90, size=1000),
    "condition": np.random.choice(["Diabetic", "Hypertensive", "Healthy"], size=1000)
}
df = pd.DataFrame(data)

# Optimize memory types
df["age"] = df["age"].astype(np.int8)  # downcast to 1 byte
df["condition"] = df["condition"].astype("category")  # map repeating strings to integers

print(df.info(memory_usage="deep"))
```

---

## ⚠️ Common Pitfalls & Debugging Tips

### 1. `SettingWithCopyWarning` in Pandas
*   **The Bug**: Modifying a subset of a DataFrame (e.g., `df[df['age'] > 50]['group'] = 'senior'`). Pandas is unsure whether this changes a view of the original DataFrame or a copy.
*   **The Fix**: Use `.loc` to modify inplace explicitly:
    ```python
    df.loc[df['age'] > 50, 'group'] = 'senior'
    ```

### 2. Iterating Over Rows (`iterrows`)
*   **The Bug**: Using `for index, row in df.iterrows():` is slow and has high overhead.
*   **The Fix**: Use vectorized operations or `.apply()` with lambda if vectorization is impossible:
    ```python
    # Avoid: df['squared'] = df['val'].apply(lambda x: x**2)
    # Better (vectorized):
    df['squared'] = df['val'] ** 2
    ```

---

## ⚡ Real-World & Project Connections

*   **Face Anti-Spoof Detection**: Real-time webcam frames are stored as 3D NumPy arrays (Height, Width, Channels) and normalized using vectorized division prior to CNN inference.
*   **Medical Condition Classification**: Loads patient diagnostic histories from structured tabular formats and cleans them using memory-minimized Pandas DataFrames.

---

## 🔗 Further Reading & Reference Links

- [NumPy Documentation & Tutorials](https://numpy.org/doc/stable/)
- [Pandas Cookbook & API Reference](https://pandas.pydata.org/docs/)
- [Modern Pandas Coding Styles](https://tomaugspurger.github.io/posts/modern-1-intro/)
