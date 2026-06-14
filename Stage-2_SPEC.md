# HFAQE Stage 2 Specification
## Embedding Layer: Mathematical Architecture, Training Dynamics & Evaluation Protocol

**Document Version:** 2.0-RC1  
**Scope:** Token Embedding Layer ONLY (HFAQE Tiered Compression)  
**Classification:** Technical Specification — Mathematical Foundations  
**Status:** Stage 2 (Post-Mortem & Redesign)

---

## 1. Forensic Analysis of Stage 1

### 1.1 The Random-Baseline Trap (Theorem 1.1)

**Observation.** Stage 1 training converged to a cross-entropy loss of $\approx 5.54$ with perplexity $\approx 255$ on a vocabulary of size $V=256$. The theoretical lower bound for a uniform random predictor is

$$\mathcal{L}_{\text{random}} = \ln V = \ln 256 \approx 5.5452.$$

The observed improvement over random is $+0.5\%$, which is statistically indistinguishable from noise.

**Definition 1.1 (Learning Signal).** A training run exhibits a *learning signal* if and only if the training loss drops below $\ln V - \delta$ for some $\delta > 0.1$ within the first epoch. Stage 1 fails this definition.

### 1.2 Diagnosis: The Quantization Deadlock

**Root Cause.** Stage 1 stored the hot tier as native `int8` arrays $(Q_H, S_H)$ and attempted to apply gradients directly to these quantized containers. Because `int8` is a discrete lattice, the Straight-Through Estimator (STE) was applied to a *decompressed* tensor that had no differentiable path back to a continuous latent state. The logged gradient norm $\|g\|_2 = 0.000$ across all steps confirms that the embedding layer received **zero effective gradient**.

**Corollary 1.2.** If the trainable parameters are discrete (int8), first-order optimization is impossible regardless of learning rate, batch size, or loss function.

### 1.3 Diagnosis: Static Tiering Causes Representation Collapse

Stage 1 assigned hot/cold tiers by a static Zipfian prior computed once at initialization. Consequently:
1. **No semantic adaptation:** A token that is rare in raw frequency but semantically pivotal (e.g., `=`, `{`, `\n`) remains cold forever, forced into a low-rank subspace of dimension $r=64$ regardless of its representational needs.
2. **Basis stagnation:** The cold basis $B$ was initialized via truncated SVD and never re-orthogonalized. Over training, $B$ became non-orthonormal, amplifying gradient noise in the cold path.
3. **Hot-cold subspace drift:** The hot tier (exact, int8) and cold tier (projected, low-rank) evolved in disconnected manifolds, producing the observed near-zero intra-class similarity ($0.0052$ for digits, $-0.0037$ for letters).

### 1.4 Diagnosis: Tautological Fidelity Metrics

Stage 1 reported *Tier Fidelity* as $\|E - \hat{E}\| = 0.000$. This metric is tautological: it measures whether dequantization inverts quantization on the **same snapshot**. It says nothing about whether the embeddings $E$ themselves encode semantic structure. A random matrix quantized and dequantized also scores $0.000$ fidelity.

---

## 2. Revised Mathematical Foundation

### 2.1 The Latent Master Manifold (Definition 2.1)

**Definition 2.1 (Latent Embedding Matrix).** Let $W \in \mathbb{R}^{V \times d}$ be the **master latent embedding matrix**. $W$ is stored in `bf16` (or `fp32` during optimizer steps) and is the **sole trainable parameter** of the embedding layer. The tiered structures $(Q_H, S_H, A, B)$ are deterministic compression artifacts of $W$:

$$\{Q_H, S_H, A, B\} = \mathcal{C}(W; K, r, B_{blk})$$

where $\mathcal{C}$ is the compression operator and $B_{blk}$ is the block size for quantization.

**Theorem 2.1 (Gradient Preservation via STE).** Let the forward pass be $x = \mathcal{C}(W)$ (decompression). In the backward pass, apply the Straight-Through Estimator:

$$\frac{\partial \mathcal{L}}{\partial W} \triangleq \frac{\partial \mathcal{L}}{\partial x} \bigg|_{x = \mathcal{C}(W)}.$$

Then for any loss $\mathcal{L}$ with $\nabla_x \mathcal{L} \neq 0$, we have $\nabla_W \mathcal{L} \neq 0$ almost everywhere.

*Proof.* STE sets the Jacobian $J_{\mathcal{C}}(W) = I$ in the backward pass. Thus $\nabla_W \mathcal{L} = \nabla_x \mathcal{L} \cdot J_{\mathcal{C}}(W) = \nabla_x \mathcal{L}$. Since the LM head and cross-entropy loss produce non-zero gradients for any imperfect prediction (which holds at initialization and throughout training), $\nabla_W \mathcal{L} \neq 0$. $\square$

**Implementation Requirement.** The optimizer must hold state (moments, etc.) for $W$ only. After each `optimizer.step()`, the compression $\mathcal{C}(W)$ is recomputed (or lazily updated) to refresh $(Q_H, S_H, A, B)$. The tiered structures are **read-only** during forward/backward.

### 2.2 Differentiable Block-Wise Quantization (Definition 2.2)

**Definition 2.2 (Block-wise Affine Quantization).** For a row $W_i$ and block index $b$ (block size $B_{blk}$), define the scale

$$s_{i,b} = \frac{\max_{j \in \text{block}_b} |W_{i,j}|}{127} + \epsilon_{guard}, \quad \epsilon_{guard} = 10^{-6}.$$

The quantized code is

$$Q_{i, bB_{blk} + j} = \text{clamp}\left(\left\lfloor \frac{W_{i, bB_{blk} + j}}{s_{i,b}} \right\rceil, -127, 127\right).$$

The dequantized vector is $\hat{W}_i = \text{Dequant}(Q_i, S_i)$. The **quantization-friendly regularizer** penalizes clipping:

$$\mathcal{L}_{quant} = \sum_{i,b} \sum_{j=0}^{B_{blk}-1} \max\left(0, \; |W_{i, bB_{blk}+j}| - 127 \cdot s_{i,b} \right)^2.$$

**Theorem 2.2 (Quantization Error Bound).** For any row $W_i$ and block $b$, the per-element error satisfies

$$|W_{i,j} - \hat{W}_{i,j}| \leq \frac{s_{i,b}}{2} \leq \frac{\max_{k \in \text{block}_b}|W_{i,k}|}{254}.$$

*Proof.* Follows from rounding to nearest integer and scale definition. $\square$

### 2.3 Dynamic Gradient-Magnitude Tier Migration (Definition 2.3)

**Definition 2.3 (Migration Score).** Let $G_i^{(t)} = \|\partial \mathcal{L} / \partial W_i^{(t)}\|_2$ be the gradient norm for token $i$ at step $t$. The *migration score* over a window of $T_{win}$ steps is

$$\mu_i = \beta \cdot \frac{f_i}{\sum_j f_j} + (1-\beta) \cdot \frac{1}{T_{win}} \sum_{\tau=t-T_{win}}^{t} G_i^{(\tau)},$$

where $f_i$ is corpus frequency and $\beta \in [0,1]$ (default $\beta = 0.3$). The hot set $\mathcal{H}$ is the top-$K$ tokens by $\mu_i$.

**Algorithm 2.1 (Tier Reallocation).** Every $T_{realloc}$ steps (default $T_{realloc} = 300$):
1. Compute $\mu_i$ for all $i \in [V]$.
2. Sort descending; select new $\mathcal{H}$.
3. For $i \in \mathcal{H}$: quantize $W_i$ into $(Q_H, S_H)$.
4. For $i \notin \mathcal{H}$: compute cold coefficients $a_i = B^\dagger W_i$ (or via least-squares if $B$ is not orthonormal).
5. Reset optimizer momentum for rows that changed tiers (optional but recommended).

**Theorem 2.3 (Migration Preserves Capacity).** Under reallocation, the expected representational capacity of the hot tier is proportional to the expected gradient magnitude of the tokens it contains. Thus, tokens that are "hard to learn" (high gradient norm) are promoted to the exact int8 tier, while easy or static tokens are demoted to the compressed cold tier.

### 2.4 Adaptive Low-Rank Basis with Orthogonal Regularization (Definition 2.4)

**Definition 2.4 (Orthonormal Basis Constraint).** The cold basis $B \in \mathbb{R}^{d \times r}$ must satisfy $B^\top B = I_r$. Enforce this via a differentiable penalty:

$$\mathcal{L}_{ortho} = \|B^\top B - I_r\|_F^2.$*

Additionally, perform a **hard re-orthogonalization** every $T_{ortho}$ steps (default $T_{ortho} = 100$):

$$B \leftarrow \text{QR}(B).Q.$$

**Definition 2.5 (Per-Token Residual Monitoring).** For each cold token $i$, the reconstruction error is

$$\epsilon_i = \|W_i - B a_i\|_2^2, \quad a_i = B^\top W_i.$$

If $\epsilon_i > \epsilon_{max}$ (default $\epsilon_{max} = 0.01 \cdot \|W_i\|^2$), the token is flagged for promotion to the hot tier at the next reallocation window.

**Theorem 2.4 (Orthogonal Basis Prevents Cold Collapse).** If $B^\top B = I_r$, then the cold reconstruction is an orthogonal projection onto a rank-$r$ subspace. The gradient w.r.t. $B$ is

$$\frac{\partial \mathcal{L}}{\partial B} = \sum_{i \in \text{cold}} \frac{\partial \mathcal{L}}{\partial x_i} a_i^\top,$$

and the condition number of the cold backward pass is $\kappa = 1$.

*Proof.* With orthonormal $B$, the projection matrix $P = BB^\top$ is idempotent. The Jacobian of the reconstruction $x_i = B a_i$ w.r.t. $B$ holding $a_i$ fixed is $J = a_i^\top \otimes I_d$. Because $B$ has orthonormal columns, the singular values of the linear map $a_i \mapsto B a_i$ are all 1. $\square$

### 2.5 Geometry-Aware Semantic Loss (Definition 2.5)

**Definition 2.5 (Token Taxonomy).** Partition the vocabulary into $C$ semantic classes (e.g., $C=6$: digits, uppercase, lowercase, punctuation, whitespace, control). Let $S_c \subset [V]$ be the set of token IDs in class $c$, and let $n_c = |S_c|$.

**Definition 2.6 (Class Centroid).** The centroid of class $c$ in embedding space is

$$\mu_c = \frac{1}{n_c} \sum_{i \in S_c} W_i.$$

**Definition 2.7 (Supervised Contrastive Semantic Loss).** For a temperature $\tau > 0$ (default $\tau = 0.05$):

$$\mathcal{L}_{semantic} = -\sum_{c=1}^{C} \sum_{i \in S_c} \log \frac{\exp(\cos(W_i, \mu_c) / \tau)}{\sum_{j=1}^{V} \exp(\cos(W_i, W_j) / \tau)}.$$

This is a **supervised InfoNCE** loss that pulls tokens toward their class centroid and pushes all other tokens away.

**Theorem 2.5 (Semantic Loss Bounds Intra-Class Variance).** Minimizing $\mathcal{L}_{semantic}$ implies that the average intra-class cosine similarity is bounded below by the inter-class cosine similarity. Formally, for any class $c$:

$$\frac{1}{n_c^2} \sum_{i,j \in S_c} \cos(W_i, W_j) \geq \frac{1}{n_c(V-n_c)} \sum_{i \in S_c} \sum_{j \notin S_c} \cos(W_i, W_j) + \Delta$$

where $\Delta \geq 0$ increases as $\mathcal{L}_{semantic}$ decreases.

*Proof sketch.* The numerator of the InfoNCE term forces $\cos(W_i, \mu_c)$ to be large. By Jensen's inequality, $\cos(W_i, \mu_c) \leq \frac{1}{n_c} \sum_{j \in S_c} \cos(W_i, W_j)$. The denominator forces all other similarities to be small. $\square$

**Definition 2.8 (Hot-Cold Alignment Loss).** To prevent the hot and cold manifolds from diverging:

$$\mathcal{L}_{align} = \|\mu_{hot} - \mu_{cold}\|_2^2,$$

where $\mu_{hot} = \frac{1}{K}\sum_{i \in \mathcal{H}} W_i$ and $\mu_{cold} = \frac{1}{V-K}\sum_{i \notin \mathcal{H}} W_i$.

### 2.6 Composite Loss Function (Definition 2.9)

**Definition 2.9 (Total Embedding Loss).**

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda_1 \mathcal{L}_{semantic} + \lambda_2 \mathcal{L}_{align} + \lambda_3 \mathcal{L}_{ortho} + \lambda_4 \mathcal{L}_{quant},$$

with recommended coefficients:
- $\lambda_1 = 0.1$ (semantic geometry)
- $\lambda_2 = 0.01$ (tier alignment)
- $\lambda_3 = 0.001$ (basis orthogonality)
- $\lambda_4 = 0.001$ (quantization friendliness)

---

## 3. Architecture Specification

### 3.1 Data Structures (Revised)

```
MasterLatent:
  W: bf16[V × d]          // Trainable master copy
  m_W: fp32[V × d]        // Adam first moment (or other optimizer state)
  v_W: fp32[V × d]        // Adam second moment

HotTier (Read-Only Cache):
  Q_H: int8[K × d]        // Quantized codes
  S_H: fp32[K × m]        // Block scales, m = ceil(d / B_blk)
  idx: dict[int → int]    // Global token ID → hot slot
  global_ids: int[K]      // Slot → global token ID

ColdTier (Read-Only Cache):
  A: bf16[(V-K) × r]      // Coefficients, row-major
  B: bf16[d × r]          // Basis, COLUMN-MAJOR (for BLIS microkernel)
  idx: dict[int → int]    // Global token ID → cold slot
  global_ids: int[V-K]    // Slot → global token ID
  residual_err: fp32[V-K] // Per-token reconstruction error

TierAllocator:
  H_set: bitset[V]        // Current hot membership
  mu: fp32[V]             // Migration scores
  win_buf: RingBuffer[fp32[V]] // Gradient norm history
```

### 3.2 Forward Pass (Algorithm 3.1)

```
Algorithm: HFAQE_FORWARD_STAGE2
Input:  token_ids[0..n-1], W, H_set, Q_H, S_H, A, B
Output: embeddings[0..n-1] each of dimension d

1.  // Lazy recompression check
2.  if step % T_realloc == 0:
3.      {Q_H, S_H, A, B, H_set} ← COMPRESS(W, K, r, B_blk)
4.  
5.  for t in token_ids:
6.      if H_set[t] == 1:
7.          slot ← idx_H[t]
8.          x ← DEQUANT_AVX512(Q_H[slot], S_H[slot])   // O(d)
9.      else:
10.         slot ← idx_C[t]
11.         a ← A[slot]                                  // bf16[r]
12.         x ← COLD_RECONSTRUCT_AVX512(B, a)            // O(d·r)
13.     embeddings.append(x)
14. return embeddings
```

**Complexity:** $O(n_H \cdot d + n_C \cdot d \cdot r + d \cdot r)$ where $n_H = |\{t \in \text{token_ids} : H_set[t]=1\}|$.

### 3.3 Backward Pass (Algorithm 3.2)

```
Algorithm: HFAQE_BACKWARD_STAGE2
Input:  dL/dx[0..n-1][0..d-1], token_ids[0..n-1], W
Output: dL/dW (accumulated into W's gradient buffer)

1.  // STE: treat decompression as identity in backward
2.  for idx, t in enumerate(token_ids):
3.      g ← dL/dx[idx]            // fp32[d]
4.      dW[t] ← dW[t] + g         // atomic add for thread safety
5.  
6.  // No gradient to Q_H, S_H, A, B (they are deterministic caches)
7.  return dW
```

**Critical Invariant:** The gradient norm reported in training logs **must** be computed as $\|dW\|_F$, not $\|dQ_H\|_F$ or $\|dA\|_F$. If $\|dW\|_F = 0$, the LM head or loss function is broken, not the embedding layer.

### 3.4 Optimizer Step & Recompression (Algorithm 3.3)

```
Algorithm: HFAQE_OPTIMIZER_STEP
Input:  W, dW, optimizer_state, lr, step
Output: Updated W, refreshed tiered caches

1.  // Standard optimizer step on latent master
2.  W ← OPTIMIZER_STEP(W, dW, optimizer_state, lr)
3.  
4.  // Periodic recompression
5.  if step % T_realloc == 0:
6.      // 1. Compute migration scores
7.      for i in [0, V):
8.          mu[i] ← beta * freq[i] + (1-beta) * grad_norm_history[i]
9.      
10.     // 2. Sort and select hot set
11.     H_indices ← argsort(mu, descending=True)[0:K]
12.     H_set ← zeros(V); H_set[H_indices] ← 1
13.     
14.     // 3. Quantize hot rows
15.     for slot, i in enumerate(H_indices):
16.         Q_H[slot], S_H[slot] ← QUANTIZE(W[i], B_blk)
17.     
18.     // 4. Orthonormalize basis (hard reset)
19.     B ← QR(B).Q
20.     
21.     // 5. Project cold rows
22.     C_indices ← [i : H_set[i] == 0]
23.     for slot, i in enumerate(C_indices):
24.         a ← B^T · W[i]          // projection onto orthonormal basis
25.         A[slot] ← a
26.         residual_err[slot] ← ‖W[i] - B·a‖²
27.     
28.     // 6. Reset gradient accumulators
29.     zero_grad(dW)
30.     // 7. Reset optimizer momentum for tier-migrated rows (optional)
31.     RESET_MOMENTUM(optimizer_state, migrated_rows)
```

### 3.5 Tier-Specific Learning Rates (Definition 3.1)

Because cold tokens are compressed through a basis, their effective gradient magnitude is attenuated by the projection. Compensate with tier-specific LR multipliers:

$$\eta_{eff}(i) = \begin{cases} \eta_{base} \cdot \gamma_{hot} & i \in \mathcal{H} \\ \eta_{base} \cdot \gamma_{cold} & i \notin \mathcal{H} \\ \eta_{base} \cdot \gamma_{basis} & \text{for basis } B \end{cases}$$

Recommended defaults: $\gamma_{hot} = 1.0$, $\gamma_{cold} = 2.0$, $\gamma_{basis} = 0.5$.

---

## 4. Training Dynamics & Convergence

### 4.1 Convergence Criteria (Hard Gates)

A Stage 2 training run is **valid** only if it satisfies all of the following within the first 3 epochs:

| Gate | Metric | Threshold | Rationale |
|------|--------|-----------|-----------|
| G1 | Gradient Norm | $\|g\|_F > 0.01$ after step 50 | Confirms STE is working |
| G2 | Loss Drop | $\mathcal{L} < \ln V - 0.3$ by epoch 2 | Confirms learning signal |
| G3 | Validation Monotonicity | $\mathcal{L}_{val}^{(e+1)} < \mathcal{L}_{val}^{(e)}$ for $e=1,2,3$ | Confirms generalization |
| G4 | Semantic NNCA | NNCA@5 > 20% by epoch 1 | Confirms geometry formation |
| G5 | Tier Balance | $0.1 < \|g_{hot}\| / \|g_{cold}\| < 10$ | Confirms both tiers train |

If any gate fails, training must be halted and the architecture bug fixed before proceeding.

### 4.2 Learning Rate Schedule (Revised)

Use a **warmup + cosine decay** schedule on $\eta_{base}$:

$$\eta_{base}(t) = \eta_{max} \cdot \min\left(\frac{t}{t_{warm}}, 1\right) \cdot \frac{1}{2}\left(1 + \cos\left(\frac{t - t_{warm}}{T_{max} - t_{warm}} \pi\right)\right),$$

with $t_{warm} = 100$ steps, $\eta_{max} = 3\times 10^{-4}$.

**Rationale:** The warmup period allows the latent master $W$ to escape the initial random quantization noise before the aggressive semantic loss pulls embeddings into class clusters.

### 4.3 Gradient Clipping (Per-Token)

Instead of global clipping, clip per-token gradients to preserve semantic loss geometry:

$$g_i \leftarrow g_i \cdot \min\left(1, \; \frac{\theta_{clip}}{\|g_i\|_2}\right), \quad \theta_{clip} = 1.0.$$

---

## 5. Evaluation Protocol (Normative)

Stage 1 evaluation was flawed because it relied on **tautological fidelity** and **perplexity-only** metrics. The Stage 2 evaluation is a **3-tier protocol** that measures geometry, downstream utility, and compression fidelity independently.

### 5.1 Tier A: Intrinsic Embedding Geometry (40 points)

These metrics test the structure of the embedding space without any downstream model.

**A.1 Nearest Neighbor Class Accuracy (NNCA) — 15 points**

**Definition 5.1.** For each token $i$, let $N_k(i)$ be its $k$ nearest neighbors by cosine similarity in the final embedding matrix $W$. Let $C(i) \in \{1, \dots, 6\}$ be the semantic class of token $i$.

$$\text{NNCA}@k = \frac{1}{V} \sum_{i=1}^{V} \frac{1}{k} \sum_{j \in N_k(i)} \mathbf{1}_{[C(j) = C(i)]}.$$

- **Random baseline** (V=256, 6 balanced classes): $\approx 16.7\%$.
- **Pass threshold:** $\geq 75\%$ for 10 points, $\geq 85\%$ for 15 points.
- **Procedure:** Compute cosine similarity matrix $S = W W^\top / (\|W_i\| \|W_j\|)$. Exclude diagonal. For each row, take top-$k$ indices. Compare classes.

**A.2 Clustering Purity — 10 points**

Run k-means with $k = C = 6$ on the embedding rows $W$. Let $\Omega_p$ be the set of points assigned to cluster $p$, and let $c_p^* = \arg\max_c |\{i \in \Omega_p : C(i) = c\}|$.

$$\text{Purity} = \frac{1}{V} \sum_{p=1}^{C} |\{i \in \Omega_p : C(i) = c_p^*\}|.$$

- **Pass threshold:** $\geq 80\%$ for 5 points, $\geq 90\%$ for 10 points.

**A.3 Embedding Utilization (Eigenvalue Entropy) — 10 points**

Let $\Sigma = \frac{1}{V} W^\top W$ with eigenvalues $\lambda_1 \geq \dots \geq \lambda_d > 0$. Define

$$p_i = \frac{\lambda_i}{\sum_j \lambda_j}, \quad U = -\frac{\sum_i p_i \ln p_i}{\ln d}.$$

- $U = 0$: all variance collapsed to one direction (pathological).
- $U = 1$: perfectly isotropic (random-like, also pathological for language).
- **Pass threshold:** $U \in [0.65, 0.95]$ for 10 points. Values outside this window indicate either collapse or lack of structure.

**A.4 Anisotropy — 5 points**

$$A = \frac{1}{V^2} \sum_{i=1}^{V} \sum_{j=1}^{V} \cos(W_i, W_j).$$

- Random embeddings: $A \approx 0$.
- Good language embeddings: $A \in [0.15, 0.45]$ (slight positive correlation is natural due to shared context).
- **Pass threshold:** $A \in [0.10, 0.50]$ for 5 points.

### 5.2 Tier B: Downstream Utility (30 points)

These metrics test whether the embeddings are useful when frozen and fed into a small downstream model.

**B.1 Token Class Linear Probe — 15 points**

Freeze $W$. Train a single `nn.Linear(d, 6)` classifier (no hidden layers) for 5 epochs on the task of predicting token class from embedding.

$$\text{Accuracy} = \frac{1}{V} \sum_{i=1}^{V} \mathbf{1}_{[\arg\max_j (W_i \cdot M_j) = C(i)]}.$$

- **Pass threshold:** $\geq 85\%$ for 10 points, $\geq 95\%$ for 15 points.
- **Rationale:** If the embedding space has semantic structure, a linear probe should separate classes easily.

**B.2 Sequence Boundary Detection — 15 points**

Freeze $W$. Construct a dataset of 2-token sequences $(t_1, t_2)$. Label $y=1$ if $t_2$ is a whitespace/control token (word boundary), else $y=0$. Train a logistic regression on $[W_{t_1}, W_{t_2}]$ (concatenated, dimension $2d$).

- **Pass threshold:** F1 $\geq 0.70$ for 10 points, F1 $\geq 0.85$ for 15 points.
- **Rationale:** Byte/char embeddings must encode boundary information to be useful for language modeling.

### 5.3 Tier C: Compression Fidelity & Efficiency (30 points)

These metrics replace the tautological Stage 1 fidelity tests.

**C.1 Hot Reconstruction RMSE — 10 points**

For hot tokens, compute the *relative* RMSE against the latent master:

$$\text{RMSE}_{hot} = \sqrt{\frac{1}{K d} \sum_{i \in \mathcal{H}} \|W_i - \hat{W}_i\|^2}, \quad \text{RelErr}_{hot} = \frac{\text{RMSE}_{hot}}{\sqrt{\frac{1}{K d} \sum_{i \in \mathcal{H}} \|W_i\|^2}}.$$

- **Pass threshold:** RelErr $< 3\times$ the theoretical bound $\frac{1}{254} \approx 0.0039$ (i.e., $< 0.012$) for 10 points.
- **Critical difference from Stage 1:** We compare against the **latent master** $W$, not against the dequantized copy of itself.

**C.2 Cold Reconstruction Relative Error — 10 points**

For cold tokens:

$$\text{RelErr}_{cold} = \sqrt{\frac{\sum_{i \notin \mathcal{H}} \|W_i - B a_i\|^2}{\sum_{i \notin \mathcal{H}} \|W_i\|^2}}.$$

- **Pass threshold:** RelErr $< 2\%$ for 10 points.
- **Additional check:** Report the 95th percentile of per-token error. No single token should have RelErr $> 10\%$.

**C.3 Throughput & Memory — 10 points**

- **Hot gather:** $\geq 10^7$ tok/s for 5 points.
- **Cold reconstruct:** $\geq 5 \times 10^5$ tok/s for 5 points.
- **Memory reduction:** $\geq 30\%$ vs. dense BF16 baseline for baseline credit.

### 5.4 Tier D: Training Health (Hard Gates, No Points)

These are pass/fail criteria that must be printed in every training log.

| ID | Metric | Target | Diagnostic |
|----|--------|--------|------------|
| D1 | Latent Gradient Norm | $> 0.01$ by step 50 | If zero, STE is broken |
| D2 | Hot/Cold Gradient Ratio | $\in [0.1, 10]$ | If $\infty$, cold path is dead |
| D3 | Loss vs. Random | $< \ln V - 0.2$ by epoch 2 | If not, no learning signal |
| D4 | Basis Condition Number | $\kappa(B) < 10$ | If $> 100$, cold collapse |
| D5 | Tier Migration Count | $> 0$ over 5 epochs | If zero, allocator is frozen |

### 5.5 Final Scoring Rubric

| Tier | Max Points | Passing Threshold |
|------|------------|-------------------|
| A — Geometry | 40 | $\geq 25$ |
| B — Downstream | 30 | $\geq 20$ |
| C — Compression | 30 | $\geq 20$ |
| **Total** | **100** | **$\geq 65$ for "PRODUCTION READY"** |
| | | **$\geq 80$ for "EXCELLENT"** |
| | | **$< 50$ for "WEAK — DO NOT DEPLOY"** |

**Stage 1 achieved $\approx 33.8/100$. Stage 2 target is $\geq 75/100$.**

---

## 6. Theoretical Guarantees Summary

| ID | Theorem / Guarantee | Stage 1 Status | Stage 2 Fix |
|----|---------------------|----------------|-------------|
| T2.1 | Gradient Preservation via STE | **Broken** (gnorm=0) | Latent master $W$ + STE |
| T2.2 | Quantization Error Bound | Held | Unchanged (good) |
| T2.3 | Dynamic Tier Migration | Missing | Gradient-magnitude allocator |
| T2.4 | Orthogonal Basis Prevents Collapse | **Broken** (B drifted) | Hard QR + $\mathcal{L}_{ortho}$ |
| T2.5 | Semantic Loss Bounds Variance | Missing | Supervised InfoNCE |
| — | Hot-Cold Alignment | Missing | $\mathcal{L}_{align}$ |
| — | Quantization Friendliness | Missing | $\mathcal{L}_{quant}$ |

---

## 7. Implementation Checklist (Stage 2)

### Core.cpp Changes
- [ ] **CRITICAL:** Add `MasterLatent` struct with `bf16[V×d]` array.
- [ ] **CRITICAL:** Change backward pass to write gradients to `dW_master`, not to `dQ_H` or `dA`.
- [ ] **CRITICAL:** Change `apply_gradients()` to update `W_master`, then call `COMPRESS()`.
- [ ] Add `TierAllocator` with migration score ring buffer.
- [ ] Implement `COMPRESS()` function: quantize hot rows, QR-reorthogonalize basis, project cold rows.
- [ ] Add `L_semantic` computation (requires token class labels).
- [ ] Add `L_align` and `L_ortho` to loss computation.
- [ ] Implement per-token gradient clipping.
- [ ] Add `RESET_MOMENTUM()` for tier-migrated rows.

### Train.cpp Changes
- [ ] Report `gnorm_master = ‖dW_master‖_F` in logs (not tiered grad norms).
- [ ] Report `gnorm_hot / gnorm_cold` ratio.
- [ ] Report `NNCA@5` and `Clustering Purity` at every validation checkpoint.
- [ ] Implement cosine LR schedule with warmup.
- [ ] Add hard gates: abort training if G1–G5 fail.

### Eval.cpp (New or Revised)
- [ ] Implement `compute_nnca()` using cosine similarity matrix.
- [ ] Implement `compute_clustering_purity()` with k-means (k=6).
- [ ] Implement `compute_utilization()` via eigenvalue entropy.
- [ ] Implement `compute_anisotropy()`.
- [ ] Implement `linear_probe_accuracy()` for token class prediction.
- [ ] Implement `boundary_detection_f1()` for sequence task.
- [ ] Compute `RelErr_hot` against `W_master` (not self).
- [ ] Compute `RelErr_cold` against `W_master`.
- [ ] Report Tier D training health metrics.

### Test.cpp (New Tests)
- [ ] **Gradient Flow Test:** After one backward step, verify `‖dW_master‖_F > 0`.
- [ ] **STE Sanity Test:** Verify that `dQ_H` and `dA` are zero (they are non-trainable caches).
- [ ] **Tier Migration Test:** After 100 steps with synthetic gradient spikes on a cold token, verify it is promoted to hot set.
- [ ] **Orthogonality Test:** After training, verify `‖B^T B - I‖_F < 0.1`.
- [ ] **Semantic Loss Test:** With synthetic class data, verify `L_semantic` decreases and NNCA increases.

---

## 8. Explicit Right Evaluation (Summary)

> **The only valid evaluation of an embedding model is a three-tier protocol that tests (1) the geometric structure of the embedding space, (2) the utility of frozen embeddings on downstream tasks, and (3) the fidelity of the compression against the latent master—not against itself.**

**Do not evaluate by:**
- Perplexity alone (it tests the LM head, not the embedding geometry).
- Tier fidelity against self (it is tautological).
- Throughput alone (it tests engineering, not learning).

**Do evaluate by:**
1. **NNCA@5** and **Clustering Purity** — these are the "accuracy" of the embedding space.
2. **Linear Probe Accuracy** — this is the "accuracy" of the representations for downstream use.
3. **Relative Reconstruction Error vs. Latent Master** — this is the true compression fidelity.
4. **Gradient Norm > 0** — this is the mandatory health gate that Stage 1 failed.

---

*End of Stage 2 Specification.*
