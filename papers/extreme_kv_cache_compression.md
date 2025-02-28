# **Lexico: Extreme KV Cache Compression via Sparse Coding Over Universal Dictionaries**

## **1. Motivation & Challenges in Existing KV Cache Compression**

Large Language Models (LLMs) rely on **KV (Key-Value) caches** to store attention-related data for efficient decoding. However, KV caches **grow with sequence length** and require significant **GPU memory**, making LLM deployment **challenging in low-memory environments**.

### **Challenges with Existing KV Cache Compression Techniques**
1. **Token Eviction Strategies** (e.g., SnapKV, PyramidKV):
   - Drop less relevant tokens to reduce memory usage.
   - **Problem**: Degrades performance on long-context tasks since crucial tokens may be removed.

2. **Quantization-Based Methods** (e.g., KIVI, ZipCache):
   - Compress KV entries into lower-bit representations (e.g., **2-bit or 4-bit** quantization).
   - **Problem**: Fixed compression rates with **upper bounds on efficiency**; struggles under **2-bit quantization**.

3. **Architectural Changes** (e.g., Grouped Query Attention):
   - Reduce KV cache size through model modifications.
   - **Problem**: Requires re-training or fine-tuning, limiting applicability to pretrained models.

### **What’s Needed?**
An **off-the-shelf**, **memory-efficient**, and **universal** KV cache compression method that:
- Retains long-context understanding.
- Outperforms quantization and eviction methods in **low-memory settings**.
- Works for any **pretrained** LLM **without retraining**.

---

## **2. Introducing Lexico**

**Lexico** is a **sparsity-based KV cache compression method** that leverages:
- **Sparse Coding with a Universal Dictionary**: Each KV vector is approximated as a **sparse linear combination** of pre-learned dictionary atoms.
- **Three-Step Compression Process**: Balances memory efficiency while preserving performance.
- **Fine-Grained Control Over Compression Rates**: Allows compression beyond **2-bit quantization**.

### **How Lexico Solves Previous Challenges**
✅ **Retains performance**: Achieves **90-95% of the original accuracy** while using **only 15-25%** of KV cache memory.
✅ **Universal & Model-Agnostic**: Works across different **prompts, tasks, and models**.
✅ **Higher Compression than Quantization**: Enables compression below **20% of the original KV size** while maintaining strong accuracy.
✅ **Scales without Batch Constraints**: Unlike quantization, its memory footprint **does not grow with batch size**.

---

## **3. Lexico’s Three-Step Compression Method**

### **Step 1: Dictionary Pretraining**  
- Train a **universal dictionary** per model using **WikiText-103**.
- The dictionary **does not depend on input data** and is **fixed**.
- **Memory-efficient**: Only requires **constant memory** (not batch-dependent).
- Implemented via **dictionary learning** methods from **compressed sensing**.

### **Step 2: Sparse Decomposition (Compression)**  
- Each KV vector is represented as a **sparse linear combination** of **pretrained dictionary atoms**.
- Uses **Orthogonal Matching Pursuit (OMP)** to approximate KV pairs with a **small reconstruction error**.
- **Key Benefit**: High compression while preserving token context.

### **Step 3: Lightweight Sparse Coefficients**  
- Store sparse coefficients in **8-bit** instead of **FP16**.
- Uses **Compressed Sparse Row (CSR) format**, which saves space.
- **Key Benefit**: Achieves **better compression rates than 2-bit quantization** without accuracy loss.

---

## **4. Implementing Lexico in Python**

### **Step 1: Train a Universal Dictionary**
```python
import numpy as np
from sklearn.decomposition import DictionaryLearning

# Load training data (e.g., WikiText-103 KV cache)
X_train = np.load("kv_cache_samples.npy")  # Shape (num_samples, kv_dim)

# Train dictionary with 4096 atoms
n_atoms = 4096
dictionary = DictionaryLearning(n_components=n_atoms, transform_algorithm='omp', n_jobs=-1)
dictionary.fit(X_train)

# Save dictionary for reuse
np.save("lexico_dictionary.npy", dictionary.components_)
```

### **Step 2: Sparse Decomposition Using OMP**
```python
from sklearn.linear_model import OrthogonalMatchingPursuit

def compress_kv(kv_vector, dictionary, sparsity=16):
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=sparsity)
    omp.fit(dictionary.T, kv_vector)
    return omp.coef_  # Sparse representation

# Load pre-trained dictionary
dictionary = np.load("lexico_dictionary.npy")

# Example KV vector (from LLM inference)
kv_vector = np.random.rand(dictionary.shape[1])  # Simulated KV vector

# Compress KV using sparse coding
sparse_kv = compress_kv(kv_vector, dictionary)
```

### **Step 3: Store and Use Compressed KV Cache**
```python
import scipy.sparse

# Store compressed KV in CSR format
compressed_kv_csr = scipy.sparse.csr_matrix(sparse_kv)

# Save to disk
scipy.sparse.save_npz("compressed_kv.npz", compressed_kv_csr)

# Load and reconstruct KV
loaded_kv_csr = scipy.sparse.load_npz("compressed_kv.npz")
reconstructed_kv = loaded_kv_csr.toarray() @ dictionary
```

---

## **5. Conclusion**
Lexico provides a **novel, efficient KV cache compression technique** that significantly reduces **memory usage** without **sacrificing performance**. Unlike quantization and eviction-based methods, it:
- **Achieves compression beyond 2-bit quantization**.
- **Retains high accuracy even in low-memory settings**.
- **Works universally across different models and tasks**.

**Future work** includes optimizing **OMP latency**, **adaptive sparsity**, and integrating **dynamic compression rates** based on token importance.

---

**GitHub Repo**: [https://github.com/krafton-ai/lexico](https://github.com/krafton-ai/lexico)  
**Paper**: [ArXiv Preprint](https://arxiv.org/abs/2412.08890)
