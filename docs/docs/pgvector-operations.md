# Vector Operations: Similarity & Distance Metrics (pgvector)

This page documents the vector distance and similarity operators supported by [pgvector](https://github.com/pgvector/pgvector), with implementation notes and short guidance on when to use each.  
Operators define what “similarity” means in your semantic search, and choosing the right one depends on your embedding model and application needs.

---

## L2 (Euclidean distance) `<->`
- **Calculation:**  
  √(Σᵢ (xᵢ - yᵢ)²)
- **More similar:** Smaller value (closer to 0 means more alike)
- **Less similar:** Larger value (further apart in space)
- **When to use:**  
  Use this when your embeddings are normalized for Euclidean geometry, and you want distance to mean literal “space apart.” Good default for most general-purpose dense embeddings.

---

## L1 (Manhattan distance) `<+>`
- **Calculation:**  
  Σᵢ |xᵢ - yᵢ|
- **More similar:** Smaller value
- **Less similar:** Larger value
- **When to use:**  
  Use when you care about “city block” distance—rare for dense semantic embeddings, but sometimes more robust to outliers and for certain scientific/engineering features.

---

## COSINE (Cosine distance) `<=>`
- **Calculation:**  
  1 - (x · y) / (||x|| ||y||)
- **More similar:** Smaller (0 = same direction)
- **Less similar:** Larger (up to 2 = opposite direction)
- **When to use:**  
  Use when the *angle* (direction) between embedding vectors is what matters (not their length), which is common in modern language models (OpenAI, Cohere, etc.).  
  **Typical default for text embeddings.**

---

## DOT_PRODUCT (Negative inner product) `<#>`
- **Calculation:**  
  -Σᵢ (xᵢ * yᵢ) — note the minus!
- **More similar:** More negative value (i.e., higher dot product in absolute terms)
- **Less similar:** Less negative or positive value
- **When to use:**  
  Dot product similarity is sometimes preferred for embeddings that aren’t normalized (length varies). Negative sign allows efficient use with pgvector’s ANN indexes. Use if your embedding provider specifically recommends dot product, or you see better retrieval results in practice.  
  **Note:** Most similar result has the most negative score (first in sorted result set).

---

## HAMMING (Bit vectors) `<~>`
- **Calculation:**  
  Number of bits that differ between x and y
- **More similar:** Smaller value (fewer differing bits)
- **Less similar:** Larger value (more bits different)
- **When to use:**  
  Only applicable to binary (0/1) embeddings. Useful for quantized or hashed representations, banded MinHash, etc.

---

## JACCARD (Bit vectors) `<%>`
- **Calculation:**  
  1 - (intersection size / union size) of nonzero bits
- **More similar:** Smaller (greater overlap)
- **Less similar:** Larger (less overlap)
- **When to use:**  
  Only for bit vectors. Use if you want to measure proportion of shared/non-shared binary features (e.g., similarity between sets, MinHash signatures).

---

**Choosing your operator:**
- **Use COSINE** for most text embeddings and models (OpenAI, BERT-family, etc.).
- **Use L2** if your application prefers geometric distance or your embedding provider recommends it.
- **Try DOT_PRODUCT** when length variation matters or you have legacy models; consult your embedding docs.
- **Use HAMMING/JACCARD** only for true binary (bit) vectors.

Check your embedding model documentation for the recommended distance function, and always validate search relevance with realistic queries!
