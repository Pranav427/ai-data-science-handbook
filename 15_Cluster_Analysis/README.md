# 📘 Clustering Techniques: Unsupervised Grouping

An engineering reference for partition-based and hierarchical clustering, evaluation metrics, and spatial grouping constraints.

---

## 💡 Engineering Intuition & Metrics

Clustering groups unlabeled data points based on feature similarity. It is highly sensitive to feature scaling and metric selection.

- **K-Means Clustering**: Partitions data into \(K\) distinct clusters by minimizing the Within-Cluster Sum of Squares (WCSS):
  \[
  \text{WCSS} = \sum_{j=1}^K \sum_{x_i \in C_j} \|x_i - \mu_j\|^2
  \]
- **Selecting \(K\)**:
  - **Elbow Method**: Plotting WCSS against \(K\) and finding the "elbow" point where the rate of decrease slows down. Can be subjective.
  - **Silhouette Score**: Measures how similar a point is to its own cluster compared to other clusters (range \([-1, 1]\)). Higher is better.
- **Hierarchical Clustering**: Builds a tree (dendrogram) of nested clusters. Useful when the underlying data has a natural taxonomic structure.

---

## 🛠 Best-Practice K-Means Clustering Example

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
import numpy as np

# Unlabeled feature set
X = np.random.randn(300, 4)

# Pipeline: Scaling is crucial since K-Means relies entirely on Euclidean distance
clustering_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42))
])

# Fit and predict labels
cluster_labels = clustering_pipeline.fit_transform(X)
predicted_labels = clustering_pipeline.predict(X)

# Compute Silhouette Score
score = silhouette_score(X, predicted_labels)
print(f"Silhouette Score for K=3: {score:.4f}")
```

---

## ⚠️ Common Pitfalls & Debugging Tips

### 1. Poor Random Initialization
*   **The Bug**: K-Means converges to different, suboptimal cluster groupings on different runs because of poor random centroid initialization.
*   **The Fix**: Use `init="k-means++"`, which spaces out the initial centroids before starting the iterative optimization process.

### 2. Failure on Non-Spherical Cluster Shapes
*   **The Bug**: K-Means splits long, crescent-shaped, or nested clusters down the middle because it assumes clusters are spherical and equal in size.
*   **The Fix**: Use density-based clustering like **DBSCAN** or **HDBSCAN** for arbitrary geometries, or use **Gaussian Mixture Models (GMM)** for ellipsoidal clusters.

---

## ⚡ Real-World & Project Connections

*   **PathPilot**: Spatial clustering (e.g. DBSCAN) is used to group latitude/longitude coordinates of historical traffic slow-downs into localized congestion zones, improving routing models.

---

## 🔗 Further Reading & Reference Links

- [Scikit-Learn Clustering Documentation](https://scikit-learn.org/stable/modules/clustering.html)
- [How to Select K in Clustering](https://towardsdatascience.com/cheat-sheet-to-find-the-optimal-number-of-clusters-8b24479e0f63)
