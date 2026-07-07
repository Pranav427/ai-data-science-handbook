# 📘 Logistic Regression: Probabilistic Classification

An engineering reference for binary classification, odds ratios, decision boundaries, and loss minimization.

---

## 💡 Engineering Intuition & Log-Loss

Logistic regression models the probability of a binary outcome. It maps arbitrary continuous values to a \([0, 1]\) range.

- **Sigmoid Function**: Transforms raw model outputs (logits) into a probability:
  \[
  \sigma(z) = \frac{1}{1 + e^{-z}}
  \]
- **Log-Loss (Binary Cross-Entropy)**: The objective function minimized during training. It heavily penalizes confident incorrect predictions:
  \[
  L = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
  \]
- **Odds Ratio**: The ratio of the probability of success to the probability of failure. Every unit increase in feature \(X_j\) multiplies the odds of the positive class by \(e^{\beta_j}\).

---

## 🛠 Best-Practice Classification Pipeline Example

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# Sample training data
X = np.random.rand(100, 5)
y = np.random.choice([0, 1], size=100)

# Pipeline: Scale features (critical for L1/L2 penalties) + fit Logistic Regression
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(penalty="l2", C=1.0, class_weight="balanced"))
])

pipeline.fit(X, y)

# Return class probabilities (inference: O(D) time)
probs = pipeline.predict_proba(X[:2])
print("Class probabilities:\n", probs)
```

---

## ⚠️ Common Pitfalls & Debugging Tips

### 1. Unscaled Features with Regularization
*   **The Bug**: Applying L1/L2 penalties (the `C` parameter in Scikit-Learn) to features with vastly different scales. Features with large ranges are penalized less than features with small ranges, corrupting the feature weights.
*   **The Fix**: Always prepend a `StandardScaler` to your logistic regression estimator in a unified Pipeline.

### 2. Confused by Imbalanced Classes
*   **The Bug**: Training a model on 99% negative class data, resulting in a model that predicts 0 for everything and achieves 99% accuracy but 0% recall.
*   **The Fix**: Use the `class_weight="balanced"` argument to scale loss penalties inversely proportional to class frequencies.

---

## ⚡ Real-World & Project Connections

*   **Face Anti-Spoof Detection**: Logistic regression is used as an ultra-fast baseline classifier on top of pre-trained face embeddings to predict whether a face is real (\(1\)) or spoofed (\(0\)).

---

## 🔗 Further Reading & Reference Links

- [Scikit-Learn Logistic Regression documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Odds Ratios and Log-Odds Explained](https://towardsdatascience.com/logistic-regression-detailed-overview-46c4af4303ba)
