# 📘 Python Basics: Practical Engineering Reference

A quick-reference guide to foundational Python concepts with an emphasis on production efficiency, memory management, and clean coding standards for AI Engineering.

---

## 💡 Engineering Intuition & Scoping

When building AI pipelines, writing raw Python code requires understanding how Python manages objects, scopes, and memory to avoid resource bloat.

- **LEGB Scoping Rule**: Python resolves variable names using the order: **L**ocal, **E**nclosing (nested functions), **G**lobal, and **B**uilt-in. Keep variables local to minimize scoping side effects and memory leaks.
- **Memory Overhead (Mutable vs. Immutable)**:
  - **Immutable** (e.g., `int`, `float`, `str`, `tuple`): Modifying these creates a new object in memory.
  - **Mutable** (e.g., `list`, `dict`, `set`): Modifying these changes the object in place. Be careful when passing mutable objects to multiple functions.
- **Generators vs. Lists**: Use generator expressions (`yield`) for processing large streams of data (e.g., loading files line-by-line) to keep the memory footprint at \(O(1)\) instead of loading the entire dataset into memory \(O(N)\).

---

## 🛠 Best-Practice Implementation Example

```python
from typing import Generator, List

# Good: Using type hinting, docstrings, and a generator to prevent memory bloat
def batch_generator(data: List[float], batch_size: int) -> Generator[List[float], None, None]:
    """
    Yields successive batches from data to save memory.
    
    Args:
        data: List of numerical features.
        batch_size: Number of items per batch.
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

# Usage
raw_features = [0.1, 0.4, 0.9, -1.2, 3.4, 8.8]
for batch in batch_generator(raw_features, batch_size=2):
    print(f"Processing batch: {batch}")
```

---

## ⚠️ Common Pitfalls & Debugging Tips

### 1. Mutable Default Arguments
*   **The Bug**: Defining a function with a default mutable argument like `def append_to(element, target=[])`. The `target` list is created once when the function is defined, not every time it's called.
*   **The Fix**: Use `None` as the default value:
    ```python
    def append_to(element, target=None):
        if target is None:
            target = []
        target.append(element)
        return target
    ```

### 2. Shallow Copy vs. Deep Copy
*   **The Bug**: Assigning `new_list = old_list` only copies the reference. Modifying `new_list` will modify `old_list`.
*   **The Fix**: Use `copy.deepcopy()` if the list contains nested mutable structures.

---

## ⚡ Real-World & Project Connections

*   **Automation Scripts**: Used in all deployment pipelines, environment setups, and batch processing operations.
*   **Payload Parsing**: In **Face Anti-Spoof Detection**, clean Python controls the frame-grabbing logic from the webcam prior to sending the matrices to the model.

---

## 🔗 Further Reading & Reference Links

- [Python Official Documentation](https://docs.python.org/3/)
- [Python LEGB Scoping Explained](https://realpython.com/python-scope-legb-rule/)
- [PEP 8 – Style Guide for Python Code](https://peps.python.org/pep-0008/)
