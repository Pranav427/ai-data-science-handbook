# 📘 Basic Statistics – II: Probability & Bayesian Inference

A quick reference guide for probability theory, conditional distributions, and Bayesian reasoning in ML engineering.

---

## 💡 Engineering Intuition & Conditional Probability

Probability theory dictates how our models handle uncertainty, decision boundaries, and class imbalances.

- **Bayes' Theorem**: Calculates the probability of an event based on prior knowledge of conditions related to the event. In machine learning, it forms the foundation of generative classifiers and Bayesian optimization:
  \[
  P(A|B) = \frac{P(B|A)P(A)}{P(B)}
  \]
- **Prior Probabilities & Class Imbalance**: If a target class is extremely rare (e.g., fraud or rare medical conditions), a model with 99% accuracy can still have a poor Positive Predictive Value (precision). We must incorporate the prior probability (\(P(\text{class})\)) to evaluate model utility.
- **Independence Assumption**: Naive Bayes classifiers assume that features are conditionally independent given the class label. While rarely true in practice, this "naive" assumption drastically reduces computational complexity and works surprisingly well for high-dimensional text classification.

---

## 🛠 Best-Practice Implementation Example

```python
# Practical implementation of Bayes' Theorem
# Scenario: Medical screening test
# Prior: Disease prevalence (P(D)) = 1% (0.01)
# Sensitivity (True Positive Rate: P(T+|D)) = 99% (0.99)
# False Positive Rate (P(T+|no D)) = 5% (0.05)

def posterior_disease_prob(prior: float, sensitivity: float, fpr: float) -> float:
    """
    Computes P(D|T+), the probability of having the disease given a positive test.
    """
    p_not_d = 1.0 - prior
    # P(T+) = P(T+|D)*P(D) + P(T+|no D)*P(no D)
    p_test_positive = (sensitivity * prior) + (fpr * p_not_d)
    
    # Bayes' Theorem
    posterior = (sensitivity * prior) / p_test_positive
    return posterior

prob = posterior_disease_prob(prior=0.01, sensitivity=0.99, fpr=0.05)
print(f"P(Disease | Positive Test) = {prob:.4f}")  # Only ~16.6% due to low base rate!
```

---

## ⚠️ Common Pitfalls & Debugging Tips

### 1. The Base Rate Fallacy
*   **The Bug**: Building a classifier for a rare event and evaluating only its precision/recall on a balanced test set. When deployed, it triggers massive numbers of false positives due to the low base rate in the real population.
*   **The Fix**: Calibrate your classifier's output probabilities to reflect the production class distribution, or adjust the classification threshold based on Bayesian decision boundaries.

### 2. Underflow in Joint Probabilities
*   **The Bug**: Multiplying many small conditional probabilities (e.g., in text classification) leads to numerical underflow (floating-point value rounds to 0.0).
*   **The Fix**: Sum the log-probabilities instead:
    ```python
    # Log-transform probability multiplications
    # P(A) * P(B) -> log(P(A)) + log(P(B))
    ```

---

## ⚡ Real-World & Project Connections

*   **Medical Condition Classification**: Bayesian updates allow updating diagnostic risk assessments as new test results (evidence) arrive, rather than treating each symptom in isolation.
*   **Face Anti-Spoof Detection**: Prior probabilities of spoofing attacks (often low in general deployments) are accounted for to minimize false positive triggers that annoy real users.

---

## 🔗 Further Reading & Reference Links

- [StatQuest: Bayes' Theorem Intuition](https://statquest.org/)
- [Naive Bayes Classification in Scikit-Learn](https://scikit-learn.org/stable/modules/naive_bayes.html)
