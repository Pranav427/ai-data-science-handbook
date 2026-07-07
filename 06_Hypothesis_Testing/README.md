# 📘 Hypothesis Testing: Experimentation & Validation

A guide to hypothesis testing, statistical experiment design, A/B test validation, and sample size power analysis.

---

## 💡 Engineering Intuition & Experimentation

Hypothesis testing provides a framework to determine if a model change, UI design, or algorithm update results in a true improvement, or if the variance is due to random noise.

- **Statistical vs. Practical Significance**: A large sample size can make a tiny, useless change (e.g., reducing page load time by 1 microsecond) statistically significant (\(p < 0.05\)). Always verify if the effect size has practical business or engineering value.
- **Type I & Type II Errors**:
  - **Type I (\(\alpha\))**: False Positive. Saying the new model is better when it isn't. (Usually capped at 5%).
  - **Type II (\(\beta\))**: False Negative. Missing a true improvement because the sample size was too small. **Power** (\(1 - \beta\)) is the probability of correctly detecting a true change (usually targeted at 80%).
- **Power Analysis**: Run a power analysis *before* starting an experiment to determine the minimum sample size required to detect a target effect size. Doing this prevents running tests too long or stopping them too early.

---

## 🛠 Best-Practice A/B Testing & Power Analysis Example

```python
from statsmodels.stats.power import TTestIndPower
import numpy as np
from scipy import stats

# 1. Determine minimum sample size needed
# Effect size (Cohen's d) = 0.2 (small change), alpha = 0.05, power = 0.80
analysis = TTestIndPower()
sample_size = analysis.solve_power(effect_size=0.2, alpha=0.05, power=0.80, ratio=1.0)
print(f"Minimum sample size per group: {int(np.ceil(sample_size))}")

# 2. Evaluate actual A/B test results (t-test)
group_A = np.random.normal(loc=12.0, scale=2.0, size=400)  # Baseline
group_B = np.random.normal(loc=12.3, scale=2.0, size=400)  # Variant

t_stat, p_val = stats.ttest_ind(group_A, group_B, equal_var=False)
print(f"t-statistic: {t_stat:.4f} | p-value: {p_val:.4f}")
if p_val < 0.05:
    print("Reject Null Hypothesis: The difference is statistically significant.")
else:
    print("Fail to Reject Null Hypothesis: The difference could be due to noise.")
```

---

## ⚠️ Common Pitfalls & Debugging Tips

### 1. "p-hacking" by Continuous Peeking
*   **The Bug**: Peeking at the p-value of an A/B test every day and stopping the experiment immediately when the p-value dips below 0.05. This dramatically inflates the Type I error rate.
*   **The Fix**: Determine the sample size beforehand, run the experiment until that size is fully reached, and only compute the p-value once at the end.

### 2. Multiple Comparison Bias
*   **The Bug**: Testing 20 different metric variants simultaneously against a control group without correcting for multiple comparisons. The chance of finding a false positive purely by chance increases exponentially.
*   **The Fix**: Apply a correction method, such as the Bonferroni correction:
    \[
    \alpha_{\text{adjusted}} = \frac{\alpha_{\text{original}}}{k}
    \]
    where \(k\) is the number of simultaneous tests.

---

## ⚡ Real-World & Project Connections

*   **PathPilot**: When deploying a new routing algorithm to production, run an A/B test comparing travel times of users routed by the new engine vs. the baseline. Use hypothesis testing to verify that routing latency improvements are statistically sound.

---

## 🔗 Further Reading & Reference Links

- [Statsmodels Power & Sample Size Calculations](https://www.statsmodels.org/stable/stats.html)
- [A/B Testing Pitfalls & Multi-testing Corrections](https://towardsdatascience.com/the-bonferroni-correction-eef8876c5267)
