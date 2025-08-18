##  Attention

The concept of attention inspired by the ability of humans to selectively pay more attention to silent details and ignore details that are less important in the moment . Having acess to all information but focusing on only the most relevant information helps to ensure that no meaningful details are lost while enabling efficient use of limited memory and time .


## How it Works ?

We have keys , value and query . These all are represented as vectors . Queries and Keys are typically represented as d_K dimensional vectors . Values are represented as d_V dimensional vectors . These dimensions represent different aspects of the information associated with each key . The attention mechanism compares each query to all the keys to determine their similarity . The similarity is often calculated using the dot product . Based on similarity scores , the model assigns a weight to each value . Higher similarity leads to a higher weight , indicating that the corresponding value is more relevant to the query . It is calculated using softmax function .  This weighted sum of values is then used as the output of the attention mechanism . The value with higher weight contribute to more to the final output .



# 🔹 Attention Mechanism

### 1. Query, Key, and Value computation
Q = X · W_Q
K = X · W_K
V = X · W_V

### 2. Score computation
S = Q · Kᵀ

### 3. Scaling
S_scaled = S / √d_k

### 4. Softmax normalization
A = softmax(S_scaled)

### 5. Weighted aggregation
O = A · V

---

### ✅ Final Compact Equation
Attention(Q, K, V) = softmax( (Q · Kᵀ) / √d_k ) · V
