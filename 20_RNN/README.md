# 📘 Recurrent Neural Networks: Sequence & Temporal Models

An engineering reference for Recurrent Neural Networks (RNNs), LSTMs, GRUs, sequential state propagation, and vanishing gradient solutions.

---

## 💡 Engineering Intuition & Gated Cells

Sequence models process inputs chronologically, passing a hidden state (memory) along the time dimension to capture contextual patterns over time.

- **Standard RNN Limitation**: During backpropagation through time (BPTT), gradients are multiplied repeatedly by the weight matrix at every time step. If the weights are small, gradients shrink exponentially, leading to the **Vanishing Gradient** problem (model forgets long-term dependencies).
- **LSTM (Long Short-Term Memory)**: Introduces a cell state (conveyor belt) and gating mechanisms to control information flow:
  - **Forget Gate (\(f_t\))**: Decides what to discard from the past.
  - **Input Gate (\(i_t\))**: Decides what new information to store.
  - **Output Gate (\(o_t\))**: Decides what to output as the hidden state.
- **GRU (Gated Recurrent Unit)**: A simplified version of LSTM. Combines the cell and hidden states, using only two gates (Reset and Update). It trains faster and uses less memory.

---

## 🛠 Best-Practice PyTorch LSTM Example

```python
import torch
import torch.nn as nn

# Define LSTM Sequence Classifier
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,   # Shape: (batch, sequence, features)
            dropout=0.3
        )
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)
        # lstm_out shape: (batch, seq_len, hidden_dim)
        # h_n shape: (num_layers, batch, hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        
        # Use final hidden state of the top LSTM layer for classification
        last_hidden = h_n[-1]
        return self.fc(last_hidden)

model = LSTMClassifier(vocab_size=1000, embed_dim=128, hidden_dim=256, num_classes=2)
dummy_input = torch.randint(0, 1000, (8, 20))  # Batch size 8, Sequence length 20
output = model(dummy_input)
print("Output Logits Shape:", output.shape)
```

---

## ⚠️ Common Pitfalls & Debugging Tips

### 1. Training Latency & Sequential Bottleneck
*   **The Bug**: Training a deep LSTM on long sequence lengths (e.g. > 1000 steps) takes days because sequential step-by-step state propagation cannot be parallelized on GPUs.
*   **The Fix**: Cap sequence lengths using truncation (`max_len`), pack padded sequences using PyTorch's `pack_padded_sequence` utility, or migrate to **Transformer** architectures (which parallelize training along the sequence dimension).

### 2. Gradient Explosion on Long Sequences
*   **The Bug**: Loss values fluctuate violently or print as `NaN` during training on long text inputs.
*   **The Fix**: Implement **Gradient Clipping** to cap gradient norms to a maximum threshold before running optimizer steps:
    ```python
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    ```

---

## ⚡ Real-World & Project Connections

*   **Medical Condition Classification**: LSTMs and GRUs are used to process longitudinal electronic health records (EHR) where a patient's historical lab values and diagnosis dates are treated as a sequence to predict onset of clinical conditions.

---

## 🔗 Further Reading & Reference Links

- [Understanding LSTMs (Colah's Blog)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [PyTorch LSTM Module Documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
