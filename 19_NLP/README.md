# 📘 Natural Language Processing: Text Vectorization & Modeling

An engineering reference for tokenization methodologies, TF-IDF, dense word embeddings, and text classification pipelines.

---

## 💡 Engineering Intuition & Text Representation

Text data is unstructured and discrete. Natural Language Processing converts text strings into numerical representations that machine learning models can process.

- **Sparse vs. Dense Vectors**:
  - **TF-IDF (Sparse)**: Maps text based on frequency. It assigns high weights to terms unique to specific documents:
    \[
    \text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{N}{\text{DF}(t)}\right)
    \]
    Fast and effective for keyword matching but ignores word order and semantic context.
  - **Embeddings (Dense)**: Projects words into continuous vector spaces where semantically similar words are close (e.g., Word2Vec, GloVe, Transformer embeddings). Captures deep context.
- **Subword Tokenization**: Splitting words into sub-components (e.g., Byte-Pair Encoding or WordPiece). This solves the Out-Of-Vocabulary (OOV) problem where the tokenizer crashes on misspelled or novel words.

---

## 🛠 Best-Practice Text Pipeline Example

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Text dataset
documents = [
    "Patient presented with severe chest pain and short breath.",
    "Follow-up visit for routine blood pressure check.",
    "Acute chest infection treated with antibiotics."
]
labels = [1, 0, 1]  # 1: Urgent, 0: Routine

# Pipeline: Convert raw text to TF-IDF features and classify
nlp_pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),  # Capture unigrams and bigrams
        max_features=5000
    )),
    ("classifier", LogisticRegression(class_weight="balanced"))
])

nlp_pipeline.fit(documents, labels)

# Test inference on a new clinical note
new_note = ["Patient reports persistent cough and mild chest tightness."]
prediction = nlp_pipeline.predict(new_note)
print(f"Predicted class: {prediction[0]} (Urgent)")
```

---

## ⚠️ Common Pitfalls & Debugging Tips

### 1. Out-of-Vocabulary (OOV) Failures
*   **The Bug**: A custom word-level tokenizer trained on clean text encounters emojis, URLs, or typos during production inference, mapping them to `<UNK>` and discarding critical information.
*   **The Fix**: Use subword tokenizers (like Hugging Face `tokenizers`) which break down unknown words into character n-grams instead of discarding them.

### 2. Slow Vector Comparison Latencies
*   **The Bug**: Running exact cosine similarity checks between a query vector and millions of document embedding vectors in memory takes seconds, blocking real-time response.
*   **The Fix**: Use Approximate Nearest Neighbors (ANN) indexing libraries (like FAISS, HNSWLib, or Milvus) to reduce lookup times from \(O(N)\) to \(O(\log N)\).

---

## ⚡ Real-World & Project Connections

*   **Medical Condition Classification**: Unstructured clinical notes, symptoms, and physician remarks are vectorized using TF-IDF and transformer embeddings to classify and predict patient diseases.

---

## 🔗 Further Reading & Reference Links

- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/chapter1/1)
- [Scikit-Learn Text Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
