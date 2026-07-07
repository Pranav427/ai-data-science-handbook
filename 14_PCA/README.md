# 📘 Principal Component Analysis (PCA): Dimensionality Reduction

An engineering reference for variance-maximizing feature projection, Singular Value Decomposition (SVD), and information retention.

---

## 💡 Engineering Intuition & SVD

PCA projects high-dimensional data onto orthogonal axes that maximize variance, reducing features while retaining information.

- **Maximizing Variance**: The first principal component (PC1) points in the direction of the highest variance. PC2 is orthogonal to PC1 and points in the direction of the next highest variance.
- **Singular Value Decomposition (SVD)**: PCA is computed by finding the eigenvectors of the covariance matrix, or more efficiently using SVD on the centered data matrix \(X\):
  \[
  X = U \Sigma V^T
  \]
  where \(V^T\) contains the principal axes (loadings).
- **Explained Variance Ratio**: Indicates the proportion of the dataset's total variance that lies along each principal component. Use this to determine how many components to keep (e.g., retaining 95% of cumulative variance).

---

## 🛠 Best-Practice PCA Pipeline Example

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# Simulate high-dimensional feature set
X = np.random.randn(500, 50)

# Pipeline: PCA requires feature scaling (otherwise variables with large ranges dominate variance)
pca_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95))  # Retain enough components to explain 95% variance
])

X_reduced = pca_pipeline.fit_transform(X)
pca_step = pca_pipeline.named_steps["pca"]

print(f"Original shape: {X.shape}")
print(f"Reduced shape: {X_reduced.shape}")
print(f"Cumulative Explained Variance: {np.sum(pca_step.explained_variance_ratio_):.4f}")
```

---

## ⚠️ Common Pitfalls & Debugging Tips

### 1. Forgetting to Standardize Features
*   **The Bug**: Applying PCA directly to unscaled data where features have different units (e.g., "Age" in tens vs. "Income" in thousands). PCA will align PC1 almost entirely with "Income", ignoring "Age".
*   **The Fix**: Always standard scale (\(z\)-score scale) variables prior to PCA.

### 2. Loss of Sparsity in Text/Categorical Data
*   **The Bug**: Running standard `PCA` on a sparse TF-IDF matrix. Centering the matrix turns all zero entries into non-zero entries, exhausting RAM.
*   **The Fix**: Use `TruncatedSVD` (also known as Latent Semantic Analysis) which operates directly on sparse matrices without centering:
    ```python
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=20)
    X_reduced = svd.fit_transform(sparse_matrix)
    ```

---

## ⚡ Real-World & Project Connections

*   **Embedding Visualization**: In NLP and Computer Vision, PCA is used to project high-dimensional embeddings (e.g. 512-dimension vectors from face models) down to 2D/3D spaces for cluster analysis.

---

## 🔗 Further Reading & Reference Links

- [Scikit-Learn PCA Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [Visual Introduction to Principal Component Analysis](https://setosa.io/ev/principal-component-analysis/)
