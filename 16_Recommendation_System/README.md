# 📘 Recommendation Systems: Collaborative & Content-Based Filtering

An engineering reference for collaborative filtering, matrix factorization, sparse metrics, and cold-start mitigations.

---

## 💡 Engineering Intuition & Matrix Factorization

Recommendation systems surface items based on user preferences and historic interactions, balancing relevance, diversity, and computation limits.

- **Collaborative vs. Content-Based**:
  - **Collaborative Filtering**: Recommends items based on actions of similar users. Prone to the **Cold Start** problem (cannot recommend new items/users with zero historic interactions).
  - **Content-Based Filtering**: Recommends items matching a user's profile metadata. Helps solve cold start but limits discovery diversity.
- **Matrix Factorization**: Decomposes a sparse user-item interaction matrix \(R\) into low-rank user matrices \(U\) and item matrices \(V\) representing latent feature dimensions.
- **Cosine Similarity**: The core similarity metric between feature embeddings:
  \[
  \text{Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}
  \]

---

## 🛠 Best-Practice Sparse Similarity Example

```python
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

# 1. Represent interactions using memory-efficient CSR (Compressed Sparse Row) matrix
# Rows: Users, Columns: Items
row_idx = np.array([0, 0, 1, 2, 2])
col_idx = np.array([0, 2, 1, 0, 2])
ratings = np.array([5.0, 4.0, 1.0, 4.5, 5.0])

# CSR matrix avoids storing zeros in memory
interaction_matrix = csr_matrix((ratings, (row_idx, col_idx)), shape=(3, 3))

# 2. Compute Item-Item Similarity matrix
item_similarity = cosine_similarity(interaction_matrix.T, dense_output=False)
print("Sparse Item-Item Similarity Matrix:\n", item_similarity.toarray())
```

---

## ⚠️ Common Pitfalls & Debugging Tips

### 1. Memory Exhaustion with Dense Arrays
*   **The Bug**: Initializing user-item matrices as standard 2D NumPy arrays (e.g. 100k users \(\times\) 10k items). The array attempts to allocate gigabytes of RAM to store mostly zeros.
*   **The Fix**: Always use SciPy's sparse matrices (like `csr_matrix` or `coo_matrix`) to store only non-zero interactions.

### 2. Popularity Bias Loop
*   **The Bug**: The recommendation engine repeatedly recommends the same top-5 most popular items to every user, failing to discover niche products and leading to feedback loops.
*   **The Fix**: Apply popularity-penalized scoring or add exploration terms (e.g., \(\epsilon\)-greedy selection) to inject randomness and diversity into recommendations.

---

## ⚡ Real-World & Project Connections

*   **PathPilot**: Collaborative and content-based recommendation techniques are used to suggest customized points-of-interest (POIs) or driving paths based on route histories of similar drivers.

---

## 🔗 Further Reading & Reference Links

- [SciPy Sparse Matrix Reference](https://docs.scipy.org/doc/scipy/reference/sparse.html)
- [Intro to Collaborative Filtering](https://realpython.com/build-recommendation-engine-collaborative-filtering/)
