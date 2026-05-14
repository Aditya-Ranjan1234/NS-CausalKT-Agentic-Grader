# NS-CausalKT: Neuro-Symbolic Causal Learning for Concept Mastery
## A Full Research Blueprint — Architecture, Math, Experiments, Benchmarks

---

## 0. TL;DR — What Is New Here?

Existing knowledge tracing (KT) models (DKT, SAINT, AKT) are **purely correlational**. They predict "will the student get question Q right?" but cannot answer:
- *Why* did the student fail concept C₃?
- *What would happen if* we taught concept C₁ before C₃?
- *Which prerequisite gap* is the root cause of a failure chain?

**NS-CausalKT** closes this gap by fusing three distinct reasoning layers into one jointly-trained system:

| Layer | Role | Mechanism |
|---|---|---|
| Neural Student Model | Latent knowledge state estimation | Transformer + GRU encoder |
| Symbolic Concept Layer | Hard prerequisite rules & ontology | Differentiable Logic / Logic Tensor Networks |
| Structural Causal Model | Intervention & counterfactual reasoning | Pearl's SCM + do-calculus |

The result is a model that is *predictively competitive* with SOTA KT models while being *causally interpretable* — a gap no existing single architecture fills.

---

## 1. Problem Formulation

### 1.1 Formal Setting

Let a student's interaction history be:
```
X = {(q₁,a₁), (q₂,a₂), ..., (qₜ,aₜ)}
```
where qᵢ ∈ Q is a question and aᵢ ∈ {0,1} is correctness.

Each question qᵢ maps to one or more **concepts** cᵢ ∈ C = {c₁,...,cₙ}.

Standard KT goal: predict P(aₜ₊₁ = 1 | X, qₜ₊₁)

**Our extended goal:** additionally compute
1. Causal effect: P(aₜ₊₁ | do(teach(cⱼ))) — what if we intervene and teach concept j?
2. Root cause: argmax_{cⱼ} ACE(cⱼ → failure chain)
3. Counterfactual: P(aₜ₊₁ = 1 | Xᶜᶠ) where Xᶜᶠ is a counterfactual history

---

## 2. Architecture: NS-CausalKT

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         NS-CausalKT                                      │
│                                                                          │
│  ┌──────────────────┐   ┌─────────────────────┐   ┌──────────────────┐  │
│  │  NEURAL STUDENT  │   │  SYMBOLIC CONCEPT   │   │  CAUSAL SCM      │  │
│  │  MODEL (NSM)     │──▶│  LAYER (SCL)        │──▶│  LAYER (CSL)     │  │
│  │                  │   │                     │   │                  │  │
│  │ Transformer +    │   │ Differentiable      │   │ Structural       │  │
│  │ GRU encoder      │   │ Logic / LTN         │   │ Causal Model     │  │
│  │                  │   │                     │   │ + do-calculus    │  │
│  │ Output: hₜ ∈ ℝᵈ │   │ Output: φₜ ∈ [0,1]ⁿ│   │ Output: ACE,     │  │
│  │ (knowledge state)│   │ (rule satisfaction) │   │ counterfactuals  │  │
│  └──────────────────┘   └─────────────────────┘   └──────────────────┘  │
│           │                       │                        │             │
│           └───────────────────────▼────────────────────────┘             │
│                         Joint Prediction Head                             │
│                    P(aₜ₊₁ | hₜ, φₜ, do-adjustments)                    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component 1 — Neural Student Model (NSM)

### 3.1 Input Encoding

For each interaction (qₜ, aₜ), create an embedding:

```
eₜ = Embed(qₜ) ⊕ Embed(aₜ) ⊕ Embed(cₜ) ⊕ Δtₜ
```

where:
- Embed(qₜ) ∈ ℝ^(d/4): question embedding
- Embed(aₜ) ∈ ℝ^(d/4): correctness embedding (binary lifted to dense)
- Embed(cₜ) ∈ ℝ^(d/4): concept embedding (GCN-encoded concept graph)
- Δtₜ ∈ ℝ^(d/4): time-since-last-attempt encoding (log-scaled)

### 3.2 Transformer Encoder

Apply multi-head self-attention:

```
Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V

H = TransformerEncoder([e₁,...,eₜ])   ∈ ℝ^(T×d)
```

Use causal masking so hₜ only attends to e₁,...,eₜ.

### 3.3 GRU Refinement

Feed transformer output through a GRU to model temporal dynamics:

```
hₜ = GRU(Hₜ, hₜ₋₁)
```

This captures **forgetting curves** (spacing effect) which attention alone underweights.

### 3.4 Concept State Projection

Project hidden state to a concept mastery vector:

```
μₜ = σ(Wc · hₜ + bc)   ∈ [0,1]ⁿ
```

where μₜ[k] ≈ P(student masters concept cₖ at time t).

---

## 4. Component 2 — Symbolic Concept Layer (SCL)

### 4.1 Knowledge Graph Construction

Build a **Concept Dependency Graph (CDG)** G = (C, E) where:
- Nodes = concepts c₁,...,cₙ
- Edge (cᵢ → cⱼ) means "mastery of cᵢ is prerequisite for cⱼ"

Source this graph from:
- ASSISTments skill tags + curriculum ontology
- EdNet skill hierarchy
- Optionally: LLM-assisted prerequisite mining (GPT-4o prompted with domain text)

### 4.2 Differentiable Logic Rules

Encode prerequisite logic as **soft rules** using Logic Tensor Networks (LTN):

**Rule R1 — Prerequisite Necessity:**
```
∀s,t: masters(s, cᵢ, t) ← masters(s, cⱼ, t) when (cⱼ → cᵢ) ∈ E
```
Meaning: if cⱼ is a prerequisite of cᵢ, then mastering cᵢ implies having mastered cⱼ.

**Rule R2 — Transfer:**
```
∀s,t: masters(s, cᵢ, t+1) ∧ related(cᵢ, cⱼ) → partial_masters(s, cⱼ, t+1)
```

**Fuzzy satisfaction (Łukasiewicz t-norm):**
```
φ(R1) = min(1, 1 - μₜ[i] + μₜ[j])   for each (cⱼ → cᵢ) ∈ E
Φₜ = (1/|E|) Σ φ(Rₑ)   ∈ [0,1]   — overall rule satisfaction
```

### 4.3 Symbolic Correction Gate

Use rule satisfaction to gate the neural concept state:

```
g = σ(Wg · [μₜ; Φₜ])
μ̃ₜ = g ⊙ μₜ + (1-g) ⊙ f_sym(μₜ, G)
```

where f_sym propagates prerequisite constraints over G (one-hop GCN pass).

---

## 5. Component 3 — Causal SCM Layer (CSL)

### 5.1 Structural Causal Model Definition

Define an SCM **M = (U, V, F, P(U))**:

- **Exogenous variables U:** student ability θ, question difficulty d_q, time-of-day τ
- **Endogenous variables V:** {cₖ_mastery}ₖ₌₁ⁿ, response aₜ
- **Structural equations F:**

```
cₖ_mastery ← f_k(parents_G(cₖ), θ, u_k)     u_k ~ N(0,1)
aₜ ← Bernoulli(σ(μ̃ₜ[cq] - d_q + ε))        ε ~ N(0,σ²)
```

### 5.2 Causal Graph Over Concepts

The causal DAG is the CDG G defined in §4.1, augmented with:
```
θ → all cₖ_mastery nodes
d_q → aₜ
τ → cₖ_mastery (circadian forgetting)
```

### 5.3 Do-Calculus Intervention

**Observational distribution:** P(aₜ₊₁ | X)

**Interventional distribution** (teach concept cⱼ):
```
P(aₜ₊₁ | do(cⱼ_mastery = 1))
```

By the backdoor adjustment formula:
```
P(aₜ₊₁ | do(cⱼ = 1)) = Σ_{θ} P(aₜ₊₁ | cⱼ=1, θ) · P(θ)
```

In our neural implementation, we apply the intervention by **clamping** μ̃ₜ[j] = 1 and propagating forward through the CDG before computing the prediction.

### 5.4 Average Causal Effect (ACE)

```
ACE(cⱼ → aₜ₊₁) = E[aₜ₊₁ | do(cⱼ=1)] - E[aₜ₊₁ | do(cⱼ=0)]
```

This is the primary signal for **root cause identification**: the concept with highest ACE for a failing student is the "root cause" of failure.

### 5.5 Counterfactual Reasoning

Given observed world (X, aₜ₊₁=0), the counterfactual "what if student had practiced cⱼ k more times?" is computed via **abduction-action-prediction**:

1. **Abduction:** infer P(U | X, aₜ₊₁=0)
2. **Action:** set do(cⱼ_mastery = f(k_extra_practices))
3. **Prediction:** compute P(aₜ₊₁ = 1 | modified SCM)

---

## 6. Mathematical Formulation — Full Model

### 6.1 Joint Prediction

```
ŷₜ₊₁ = σ(Wp · [hₜ; μ̃ₜ; ACE_vec] + bp)
```

where ACE_vec is the top-k ACE scores for the current question's concepts.

### 6.2 Loss Functions

**L1 — Binary Cross-Entropy (prediction):**
```
L_pred = -Σₜ [aₜ log ŷₜ + (1-aₜ) log(1-ŷₜ)]
```

**L2 — Symbolic Consistency Loss:**
```
L_sym = -Σ_{(cⱼ→cᵢ)∈E} [μ̃ₜ[j] · log μ̃ₜ[i] + (1-μ̃ₜ[j]) · log(1-μ̃ₜ[i])]
```
(Forces prerequisite satisfaction: mastering cᵢ should correlate with mastering prerequisite cⱼ)

**L3 — Causal Regularization:**
```
L_causal = Σ_k max(0, μ̃ₜ[k] - max_{j: (cⱼ→cₖ)∈E} μ̃ₜ[j] - γ)
```
(Penalizes impossibly high mastery of concept k if all its prerequisites have low mastery)

**L4 — Temporal Smoothness (anti-catastrophic-forgetting):**
```
L_smooth = Σₜ ||μ̃ₜ - μ̃ₜ₋₁||²₂
```
(Knowledge state should not jump discontinuously without practice)

**Total Objective:**
```
L_total = L_pred + λ₁·L_sym + λ₂·L_causal + λ₃·L_smooth
```

Hyperparameter search: λ₁ ∈ {0.1, 0.5, 1.0}, λ₂ ∈ {0.1, 0.5}, λ₃ ∈ {0.01, 0.1}

### 6.3 Optimization

- **Optimizer:** AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler:** Cosine annealing with warm restarts (T₀=10 epochs)
- **Gradient clipping:** ||∇||₂ ≤ 1.0
- **Batch size:** 64 sequences (padded to max_len=200)

---

## 7. Causal Graph Formulation

### 7.1 CDG Example (Math Domain)

```
Arithmetic → Fractions → Algebra → Calculus
     ↓                       ↓
  Decimals           Linear Equations
                             ↓
                     Quadratic Equations
```

### 7.2 Full DAG for SCM (Variables)

```
θ (student ability) ──────────────────────────────────────┐
         │                                                 │
         ▼                                                 ▼
c₁_mastery → c₂_mastery → ... → cₙ_mastery             aₜ₊₁
         ↑                           │                    ↑
d_q (difficulty) ────────────────────┘                    │
                                                          │
τ (time effect) ──────────────────────────────────────────┘
```

### 7.3 d-Separation for Causal Identification

To identify causal effect of cⱼ on aₜ₊₁:
- Backdoor criterion satisfied by conditioning on {θ, d_q}
- Front-door criterion applicable when θ is unobserved (via proxy variables)

---

## 8. Dataset & Experimental Setup

### 8.1 Datasets

| Dataset | Students | Questions | Concepts | Interactions | Notes |
|---|---|---|---|---|---|
| ASSISTments 2009 (ASSIST09) | ~4,151 | 17,751 | 123 | 346,860 | Primary benchmark |
| ASSISTments 2012 (ASSIST12) | ~46,674 | 179,999 | 265 | 2.7M | Scale test |
| EdNet-KT1 | 784,309 | 13,169 | 188 | 95.5M (sample 100K students) | Generalization |
| Statics2011 | 333 | 1,223 | 85 | 189,297 | Small, high-quality |

### 8.2 Data Preprocessing

1. **Sequence truncation:** max length = 200 interactions per student; pad with mask tokens
2. **Concept mapping:** map each question to exactly one primary concept (multi-concept → duplicate rows, one per concept)
3. **Temporal features:** compute Δt = time since last attempt on same concept; log-scale normalize
4. **Train/val/test split:** 70% / 10% / 20% **student-level split** (not interaction-level, to avoid leakage)
5. **Cold-start filter:** remove students with <10 interactions from test set
6. **CDG construction:**
   - ASSIST09: use published skill hierarchy
   - EdNet: extract from platform's tag taxonomy
   - Supplement with prerequisite mining: cosine similarity of concept embeddings from pre-trained LM (threshold > 0.7)

### 8.3 Feature Set Per Interaction

```
[question_id, concept_id, response, time_delta, attempt_num_on_concept,
 hint_used (ASSIST only), n_correct_prev_5, n_attempts_prev_5]
```

---

## 9. Baselines

| Model | Type | Reference |
|---|---|---|
| **BKT** | HMM probabilistic | Corbett & Anderson 1994 |
| **DKT** | LSTM neural | Piech et al. 2015 |
| **DKVMN** | Memory-augmented NN | Zhang et al. 2017 |
| **SAKT** | Self-attention KT | Pandey & Karypis 2019 |
| **AKT** | Transformer + monotonic attention | Ghosh et al. 2020 |
| **SAINT+** | Transformer (temporal) | Shin et al. 2021 |
| **SimpleKT** | Simplified transformer | Liu et al. 2023 |
| **CausalKT (ablated)** | DKT + causal, no symbolic | Ours minus SCL |
| **SymKT (ablated)** | DKT + symbolic, no causal | Ours minus CSL |
| **NS-CausalKT (full)** | Ours | — |

---

## 10. Evaluation Metrics

### 10.1 Learning Metrics (Standard KT)

| Metric | Formula | Notes |
|---|---|---|
| **AUC-ROC** | Area under ROC curve | Primary ranking metric |
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Threshold at 0.5 |
| **RMSE** | √(Σ(ŷ-a)²/N) | Calibration quality |
| **Log-loss** | -Σ[a·log ŷ + (1-a)·log(1-ŷ)] / N | |

### 10.2 Causal Metrics (Novel Contributions)

**Intervention Correctness (IC@k):**
Simulate "teach concept cⱼ" by giving student k=5 synthetic correct answers for cⱼ, then measure if predicted performance on dependent concepts cᵢ (where cⱼ → cᵢ ∈ E) improves. IC@k = fraction of cases where prediction improves as expected.

**Root Cause Accuracy (RCA):**
Construct synthetic students with known planted prerequisite gaps. Measure accuracy of NS-CausalKT's ACE-based root cause identification vs. ground truth.
```
RCA = (# correctly identified root causes) / (# students with planted gaps)
```

**Counterfactual Validity (CFV):**
Using ASSISTments' hint data as natural experiments: students who received hints approximate do(see_hint=1). Measure correlation between model's counterfactual prediction and observed gain from hints.

### 10.3 Educational Impact Metrics

**Learning Gain Improvement (LGI):**
Simulate adaptive tutoring: use ACE rankings to recommend next concept, vs. random ordering vs. DKT-based ordering. Measure cumulative accuracy gain over 20 steps.

**Concept Coverage Rate (CCR):**
Fraction of concepts where student reaches mastery threshold (μ̃ₜ[k] ≥ 0.8) within T steps under ACE-guided recommendation vs. baselines.

---

## 11. Ablation Studies

| Configuration | What Is Removed | Expected Effect |
|---|---|---|
| **Full NS-CausalKT** | Nothing | Best overall |
| **w/o Causal Layer** | CSL (no SCM, no do-calculus) | AUC ≈ same, RCA drops ~30%, IC@k drops |
| **w/o Symbolic Layer** | SCL (no CDG, no LTN rules) | AUC drops 1-2%, L_sym=0, less interpretable |
| **w/o GRU** | Replace GRU with mean pooling | AUC drops, temporal modeling lost |
| **w/o L_causal** | Set λ₂=0 | Causal constraints violated more, RCA drops |
| **w/o L_sym** | Set λ₁=0 | CDG constraints not enforced |
| **Hard rules only** | Replace LTN with boolean constraints | AUC drops (non-differentiable) |
| **Random CDG** | Shuffle CDG edges | Validates that CDG structure matters |

Report all ablations on ASSIST09 val set. Use Wilcoxon signed-rank test for significance.

---

## 12. Interpretability

### 12.1 Which Concept Caused Failure?

For a student failing question q (concept cq):

```
Root_Cause(student, q) = argmax_{cⱼ ∈ ancestors(cq, G)} ACE(cⱼ → aₜ₊₁)
```

Output to teacher: "Student is likely failing *Quadratic Equations* because of a gap in *Linear Equations* (ACE=0.42)."

### 12.2 Attention Visualization

Plot Transformer attention weights heatmap over concept history. Highlight which past practice events most influenced the current knowledge state.

### 12.3 Concept Mastery Timeline

Plot μ̃ₜ[k] over time for each concept k. Overlay question attempts and ACE-ranked recommendations.

### 12.4 Counterfactual Dashboard (Real Use Case)

Teacher input: "What if student had practiced Fractions 3 more times before this test?"
System computes counterfactual μ̃ₜᶜᶠ and outputs: "Predicted score would improve from 62% → 78%."

---

## 13. Generalization Study

### 13.1 Cross-Dataset Transfer

- Train on ASSIST09, test on Statics2011 (zero-shot concept alignment via embedding similarity)
- Train on ASSIST12, test on EdNet-KT1

Measure AUC degradation. Compare to DKT and AKT under same protocol.

### 13.2 Unseen Concepts (Zero-Shot KT)

Remove 20% of concepts from training CDG. At test time, embed unseen concepts via GCN transductive inference (using their position in the CDG relative to seen nodes).

Metric: AUC on questions from unseen concepts only.

### 13.3 Cold-Start Students

Evaluate on students with only 5-20 interactions. The symbolic CDG acts as prior knowledge → expect NS-CausalKT to outperform purely data-driven models (DKT, SAINT) in cold-start regime.

---

## 14. Error Analysis

### 14.1 Failure Mode Taxonomy

| Failure Type | Cause | Mitigation |
|---|---|---|
| **Concept ambiguity** | Question maps to multiple concepts | Multi-label concept attention |
| **Noisy CDG edges** | LLM-mined prerequisites wrong | Human validation loop |
| **Causal non-stationarity** | CDG changes across curricula | Domain-adaptive CDG fine-tuning |
| **Sparse students** | <10 interactions → unreliable μ̃ₜ | Stronger symbolic prior weight (λ₁↑) |
| **Long dependencies** | Prerequisite chains >4 hops | Increase GCN depth |

### 14.2 Quantitative Error Analysis

- Plot AUC vs. number of training interactions (learning curve)
- Plot RCA vs. prerequisite chain depth
- Calibration plot: predicted P(correct) vs. actual accuracy (binned)
- Confusion matrix for root cause identification

---

## 15. Step-by-Step Build Plan

### Phase 1 — Data Infrastructure (Week 1–2)

```
1. Download ASSIST09, ASSIST12, EdNet-KT1 from official sources
2. Implement preprocessing pipeline (Python, pandas + PyTorch Dataset)
3. Build CDG from skill tags + LLM prerequisite mining
4. Unit test: verify no train/test student overlap
5. Compute dataset statistics, verify class balance
```

### Phase 2 — NSM Implementation (Week 3–4)

```
1. Implement input embeddings (question, concept, response, time)
2. Build Transformer encoder (d=256, 4 heads, 2 layers)
3. Add GRU refinement layer
4. Implement concept projection head (μₜ)
5. Train DKT baseline to verify data pipeline is correct
6. Target: AUC > 0.82 on ASSIST09 with NSM alone
```

### Phase 3 — SCL Implementation (Week 5–6)

```
1. Implement CDG as PyTorch Geometric graph
2. Implement LTN-style fuzzy rules (vectorized Łukasiewicz t-norm)
3. Implement symbolic correction gate
4. Add L_sym to training loop
5. Ablation: NSM+SCL vs. NSM alone
6. Verify CDG constraints are being respected (monitor L_sym during training)
```

### Phase 4 — CSL Implementation (Week 7–9)

```
1. Define SCM variables and structural equations
2. Implement do-calculus intervention (concept clamping + forward pass)
3. Implement ACE computation (vectorized over all concepts)
4. Implement counterfactual reasoning (abduction-action-prediction)
5. Add L_causal to training loop
6. Implement RCA evaluation harness with synthetic planted gaps
```

### Phase 5 — Joint Training & Tuning (Week 10–11)

```
1. Joint train all three components end-to-end
2. Grid search λ₁, λ₂, λ₃
3. Learning rate sweep
4. Add gradient clipping, mixed precision (fp16)
5. Track all metrics on val set; use early stopping on AUC
```

### Phase 6 — Baselines & Benchmarks (Week 12–13)

```
1. Implement/reproduce: BKT, DKT, DKVMN, SAKT, AKT, SAINT+
2. Use pyKT library where available (https://github.com/pykt-team/pykt-toolkit)
3. Run all models on ASSIST09, ASSIST12, EdNet
4. Statistical significance tests (Wilcoxon)
5. Ablation table
```

### Phase 7 — Causal Evaluation (Week 14)

```
1. Construct synthetic planted-gap dataset for RCA
2. Compute IC@k on hint-usage subset of ASSISTments
3. Compute CFV using hint natural experiments
4. LGI simulation: adaptive tutoring trial
```

### Phase 8 — Interpretability & Visualization (Week 15)

```
1. Build attention heatmap visualizer
2. Build concept mastery timeline plotter
3. Implement root cause explanation renderer
4. Case studies: 5 student profiles with narrative explanations
```

### Phase 9 — Generalization & Error Analysis (Week 16)

```
1. Cross-dataset experiments
2. Unseen concept zero-shot evaluation
3. Cold-start analysis
4. Error taxonomy, calibration plots
```

### Phase 10 — Writing & Submission (Week 17–18)

```
Target venues: AAAI 2026, NeurIPS 2025 workshops (AIED, CausalML),
               EDM 2026, AIED 2026, IJAIED journal
```

---

## 16. Technology Stack

```python
# Core framework
torch >= 2.3.0
torch-geometric >= 2.5.0  # CDG GCN layers

# KT baselines
pykt-toolkit  # pykt-team/pykt-toolkit (DKT, AKT, SAINT+, etc.)

# Causal inference
dowhy >= 0.11             # SCM + do-calculus
econml                    # Causal effect estimation
pgmpy                     # Bayesian network / DAG tools

# Symbolic / LTN
ltn (LTN-pytorch)         # Logic Tensor Networks

# Experiment tracking
wandb                     # Metrics, hyperparameter sweeps

# Data
pandas, numpy
torch.utils.data.Dataset  # Custom KT dataset loader

# Visualization
matplotlib, seaborn       # Mastery timelines, attention heatmaps
networkx                  # CDG visualization
```

---

## 17. Expected Results (Hypothesis)

| Metric | DKT | AKT | NS-CausalKT (ours) |
|---|---|---|---|
| AUC (ASSIST09) | 0.821 | 0.849 | **~0.857** |
| AUC (EdNet) | 0.793 | 0.819 | **~0.826** |
| RCA | N/A | N/A | **~0.71** |
| IC@5 | N/A | N/A | **~0.68** |
| LGI (+Δ ACC) | baseline | +2.1% | **+5.3%** |

Note: AUC improvements beyond SOTA are modest (+0.5-1%) — the main contribution is the *causal capability*, not just raw prediction.

---

## 18. Connection to SDG-4 (Quality Education)

NS-CausalKT directly enables:

1. **Personalized Learning:** ACE rankings generate per-student concept learning queues, not one-size-fits-all curricula.
2. **Adaptive Tutoring:** Counterfactual reasoning lets an ITS decide "teach cⱼ now" by simulating outcomes before committing.
3. **Teacher Decision Support:** Root cause explanations translate model predictions into actionable pedagogical interventions ("this student needs more fraction practice, not more algebra").
4. **Equity:** Cold-start symbolic prior helps low-data students (common in under-resourced settings) get better recommendations.
5. **Curriculum Design:** Causal graph reveals which prerequisite edges matter most, informing curriculum designers.

---

## 19. Novelty Summary (For Paper Framing)

> **NS-CausalKT is the first knowledge tracing system to jointly train a Transformer-based neural student model, a differentiable Logic Tensor Network encoding curriculum prerequisites, and a Pearl-style Structural Causal Model supporting do-calculus interventions — producing a single architecture that is simultaneously predictively competitive, causally identifiable, and symbolically interpretable.**

Prior art gaps this closes:
- DKT/AKT: predictive but causally opaque
- BKT: interpretable but no causal intervention support
- Causal-DKVMN (Zhu et al. 2024): causal but no symbolic prerequisite layer
- Hooshyar et al. 2024: neuro-symbolic but no SCM or do-calculus

---

*Document prepared for research planning. All cited baselines should be reproduced from original codebases for fair comparison.*
