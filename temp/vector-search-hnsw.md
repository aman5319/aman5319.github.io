---
draft: true
---

https://github.com/facebookresearch/faiss
https://web.stanford.edu/class/cs246/slides/11-graphs.pdf


HNSW Algorithms

Good question — and you’re exactly right: **HNSW (Hierarchical Navigable Small World)** is one of the best-performing **ANN (Approximate Nearest Neighbor)** algorithms for large-scale vector search today.

Let’s unpack this systematically.

---

## 🔍 Background: What ANN Algorithms Do

ANN algorithms trade **a bit of accuracy** for **a big speedup and lower memory** in finding the nearest neighbors in high-dimensional vector spaces.
They aim to approximate the brute-force exact search (which is (O(ND)), where (N) is number of vectors, (D) is dimension).

Different ANN methods achieve this trade-off using different principles — partitioning space, quantizing vectors, or building graph structures.

---

## 🧭 Major Categories of ANN Algorithms

Broadly, **ANN algorithms** fall into **four main families**, each with several variations and hybrids.

---

### **1. Tree-based Methods**

Use spatial partitioning structures.

#### Examples:

* **KD-Tree / Ball-Tree / VP-Tree**

  * Partition space along data dimensions.
  * Work decently for *low-dimensional* vectors (say < 30D).
  * Poor scaling in high dimensions (curse of dimensionality).
* **Annoy (Spotify)**

  * Uses multiple random projection trees.
  * Very memory-efficient and simple.
  * Slower build, but fast queries with reasonable recall.

#### Typical Libraries:

* Annoy
* ScaNN (uses tree + quantization hybrid)

---

### **2. Hash-based Methods**

Use **Locality Sensitive Hashing (LSH)** to bucket similar vectors into the same hash bins with high probability.

#### Variants:

* **E2LSH (Euclidean LSH)**
  Hash functions based on random projections.
* **SimHash**
  For cosine similarity (binary signatures).
* **Multi-Probe LSH**
  Improves recall by probing nearby buckets.

#### Pros/Cons:

* Theoretically elegant, sublinear query time.
* Requires large memory; recall can drop for dense, high-dimensional vectors.

#### Typical Libraries:

* FAISS (LSH-based index)
* FALCONN

---

### **3. Quantization-based Methods**

Compress vectors into short codes, enabling fast distance computation.

#### Variants:

* **Product Quantization (PQ)**

  * Split vector into subspaces, quantize each independently.
  * Used heavily in **FAISS**.
* **Optimized Product Quantization (OPQ)**

  * Adds rotation to minimize quantization error.
* **Additive Quantization (AQ)**, **Residual Quantization (RQ)**

  * Combine multiple codebooks to approximate vectors.

#### Pros/Cons:

* Compact memory footprint.
* Good trade-off between accuracy and efficiency.
* Requires offline training to learn codebooks.

#### Typical Libraries:

* **FAISS (Facebook AI Similarity Search)** — PQ, OPQ, IVF-PQ, IVF-HNSW hybrids.
* **ScaNN (Google)** — asymmetric hashing + quantization.

---

### **4. Graph-based Methods**

Represent dataset as a proximity graph where each node is connected to its nearest neighbors.
Search traverses the graph using greedy or heuristic walks.

#### Variants:

* **NSW (Navigable Small World)**
* **HNSW (Hierarchical NSW)** — multi-layer graph; near-SOTA.
* **Vamana / DiskANN (Microsoft)** — optimized for disk-based storage.
* **SPTAG (Microsoft)** — another graph-based method.
* **ONNG (Optimized Navigable Graph)** — more compact, slightly faster build.

#### Pros/Cons:

* Excellent recall-speed trade-off.
* Handles large, dynamic datasets well (especially HNSW).
* Build time and memory overhead can be high.

#### Typical Libraries:

* **HNSWlib**
* **FAISS (has HNSW backend)**
* **NMSLIB**
* **SPTAG**, **DiskANN**

---

## ⚙️ 5th (Emerging) Hybrid or Learned Indexes

There’s a newer trend of combining or learning search structures:

* **Hybrid approaches:** IVF (coarse quantization) + HNSW graph (FAISS IVF-HNSW).
* **Learned Indexes:** models trained to predict candidate regions for search (e.g., DeepANN, LIRA, learned PQ).
* **Vector Databases:** Milvus, Pinecone, Weaviate — typically use HNSW, IVF, or PQ internally.

---

## 🧩 Summary Table

| Category       | Example           | Key Idea                 | Strength           | Weakness              |
| -------------- | ----------------- | ------------------------ | ------------------ | --------------------- |
| Tree-based     | Annoy             | Space partitioning       | Simple, low memory | Poor for high dims    |
| Hash-based     | LSH               | Similar items hash close | Fast random access | Recall, high memory   |
| Quantization   | PQ, OPQ           | Vector compression       | Efficient memory   | Training, approximate |
| Graph-based    | HNSW              | Proximity graph search   | High recall-speed  | Heavy build cost      |
| Hybrid/Learned | IVF+HNSW, DeepANN | Combine methods          | Balanced trade-off | Complex tuning        |

---

If you want, I can show **a taxonomy diagram or decision guide** (e.g., which algorithm to use based on data size, dimensionality, latency budget, etc.).
Would you like that?
