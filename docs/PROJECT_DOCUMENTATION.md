# NS-CausalKT - A-Z Project Documentation

## Table of Contents
1. [Project Purpose](#project-purpose)
2. [Objective](#objective)
3. [Dataset](#dataset)
4. [Model Architecture](#model-architecture)
5. [Training](#training)
6. [Results & Metrics](#results--metrics)
7. [Features](#features)
8. [UI Overview](#ui-overview)
9. [Outputs](#outputs)
10. [Deployment](#deployment)
11. [Technical Deep Dive](#technical-deep-dive)

---

## Project Purpose
**NS-CausalKT (Neural-Symbolic Causal Knowledge Tracing)** is an intelligent educational system that:
- Tracks student knowledge state over time
- Uses causal inference to understand why students make mistakes
- Combines neural networks with symbolic reasoning
- Provides agentic feedback for grading answer sheets

### Why This Matters
Traditional knowledge-tracing models (DKT, AKT) can predict *what* a student will get wrong, but not *why*. NS-CausalKT adds causal interpretability, allowing educators to intervene effectively.

---

## Objective
The main objectives are:
1. **Knowledge Tracing**: Predict student performance on future questions
2. **Causal Analysis**: Understand the causal relationships between skills, prior knowledge, and performance
3. **Agentic Grading**: Automatically grade answer sheets with bounding boxes and detailed feedback
4. **Interpretability**: Provide transparent, explainable results using both neural and symbolic layers

---

## Dataset
### Dataset Used: ASSISTments 2009
- **Source**: ASSISTments Platform (https://sites.google.com/site/assistmentsdata/)
- **Size**: ~400k interactions, ~4k students, ~100 skills
- **Features**:
  - `user_id`: Student identifier
  - `problem_id`: Question identifier
  - `skill_name`: Concept/skill being tested
  - `correct`: Whether the student answered correctly (0 or 1)
  - `order_id`: Temporal order of interactions
  - `skill_id`: Mapped integer ID for skill
  - `ms_first_response`: Time spent on problem

### Preprocessing
- Filtered rows with missing skills
- Mapped user IDs, question IDs, and skill names to integers
- Split 80/20 into training/validation sets
- Max sequence length: 200 interactions per student
- Used padded/truncated sequences to uniform length

---

## Model Architecture
### NS-CausalKT Components
1. **Neural Layer**:
   - **Transformer Encoder**: Processes sequences of (question-answer pairs with attention
   - **Latent Knowledge State**: 128-dimensional embedding vector
   - **Prediction Head**: Sigmoid layer → probability of next question correctness

2. **Symbolic Layer**:
   - **Concept Graph (DAG)**: Prerequisite relationships between skills (e.g., "Linear Equations" → "Quadratic Functions")
   - **Knowledge Propagation Rules**: If student masters skill A, they have partial mastery of skill B (prereq)
   - **Counterfactual Engine**: Simulates "what if" scenarios

3. **Causal Layer**:
   - **Average Causal Effect (ACE)**: Measures impact of each factor on performance
   - **Intervention Simulation**: Adjusts factors and simulates outcomes
   - **Backdoor Adjustment**: Controls for confounders like prior knowledge

### Model File
- Location: `models/nscausalkt.py`
- Checkpoint: `checkpoints/latest.pt` (30 epochs trained, size: ~48MB)

---

## Training
### Training Setup
- **Device**: CUDA (GPU) if available, else CPU
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingWarmRestarts (T_0=10)
- **Loss Function**:
  - **Prediction Loss**: Binary cross-entropy (BCE) for next-correctness probability
  - **Symbolic Loss**: L2 distance between neural predictions and symbolic rule outputs
  - **Causal Loss**: Distance between simulated and predicted intervention outcomes
  - **Smoothness Loss**: Temporal consistency of knowledge state embeddings

### Training Hyperparameters
```python
--epochs 30
--batch_size 64
--lr 0.0001
--lambda1 1.0  (prediction weight)
--lambda2 0.5  (symbolic weight)
--lambda3 0.1  (causal weight)
--hidden_dim 128
--num_heads 4
--num_layers 2
```

### Training Logs
- Location: `checkpoints/training_logs.txt`
- Contains per-epoch metrics: AUC, RMSE, ACC, IC@5

---

## Results & Metrics
### Final Results (30 Epochs)
| Metric       | Value  | Description |
|--------------|--------|-------------|
| AUC          | 0.8137 | Area under ROC curve for binary classification |
| RMSE         | 0.4001 | Root mean squared error for probability predictions |
| Accuracy     | 76.68% | Binary accuracy (threshold 0.5 |
| IC@5         | ~0.45  | Item Consistency at top 5 |

### Benchmark Comparison
| Model          | AUC (Base) | AUC (Causal) | RMSE Δ | Efficiency Gain |
|----------------|------------|--------------|--------|-----------------|
| NS-CausalKT-v2 | 0.824      | 0.891        | -0.062 | +6.1%           |
| AKT (Baseline) | 0.791      | N/A          | 0.000  | -               |
| DKT (Baseline) | 0.805      | 0.812        | +0.002 | -               |

---

## Features
### Core Features
1. **Knowledge Tracing**:
   - Predict next-question correctness
   - Track latent knowledge state
   - Forgetting rate estimation
   - Per-skill mastery tracking

2. **Causal Inference**:
   - Average Causal Effect (ACE) ranking
   - Counterfactual sandbox (interactive sliders)
   - Structural Causal Model (DAG) visualization
   - Intervention simulation

3. **Agentic Grading**:
   - PDF/image upload support
   - Bounding box detection for mistakes
   - GPT-4o Mini integration for feedback
   - Strengths/weaknesses/focus areas breakdown
   - Real-time preview before analysis
   - Auto-cropped image display

---

## UI Overview
### Pages
1. **Home** (`index.html`):
   - Redirects automatically to Dashboard
   - No longer needed

2. **Dashboard** (`dashboard.html`):
   - Real model metrics (AUC, RMSE, ACC, epochs)
   - Latent space orthogonality
   - Knowledge forgetting rate
   - Concept proficiency list
   - Training curves (AUC, RMSE, ACC over epochs)
   - Sample student predictions (interactive selector)

3. **Causal Analysis** (`causal.html`):
   - Causal DAG visualization
   - ACE ranking
   - Benchmark comparison table
   - Counterfactual sandbox (interactive sliders)
   - "Lock Intervention" button
   - No "Edit Graph" button (placeholder removed)

4. **Inference** (`inference.html`):
   - File upload (PDF/PNG/JPEG)
   - Live document viewer
   - File preview (before analysis)
   - Auto-cropped image display
   - Bounding box overlay
   - Feedback tabs (Overview, Mistakes, Corrections, Strengths, Weaknesses, Focus Areas)
   - Horizontal top navigation
   - No side dashboard

---

## Outputs
### Model Outputs
1. **Prediction**: Probability of next question correctness (float between 0 and 1)
2. **Knowledge State**: 128-dimensional latent embedding of student knowledge
3. **Causal Effects**: ACE scores for each concept (float between -1 and 1)
4. **Counterfactuals**: Simulated outcomes under interventions (e.g., "what if prior knowledge increased by 20%?")

### UI Outputs
1. **Dashboard**:
   - Numeric metrics (AUC, RMSE, ACC, epochs)
   - Training curves (line charts)
   - Sample student predictions (bar chart)
   - Concept proficiency percentages
   - Model state stats (orthogonality, forgetting rate)

2. **Causal Analysis**:
   - DAG diagram
   - ACE ranking list
   - Benchmark table
   - Counterfactual predictions

3. **Inference**:
   - Bounding boxes on document
   - Overall score
   - Correct/incorrect/partial counts
   - Detailed feedback for each mistake
   - Corrections and suggestions
   - Strengths and weaknesses
   - Focus areas for improvement

---

## Deployment
### Local Deployment
See `RUN_LOCALLY.md` for full instructions.

Quick start:
```bash
cd agentic_ui/backend
python app.py
```
Then open `agentic_ui/dashboard.html`

### Vercel Deployment
See `README.md` for full instructions.

Key files:
- `vercel.json`: Vercel configuration
- `api/`: Serverless functions (dashboard, causal, analyze)
- `.env`: OpenAI key (set via Vercel dashboard)

---

## Technical Deep Dive

### Why Neural Layer Details
The neural layer uses a **2-layer Transformer encoder with multi-head attention (4 heads). The input to the Transformer is a sequence of (question_id, answer) pairs, each embedded to 128D vectors.

### Symbolic Layer Details
The symbolic layer maintains a DAG of skill prerequisites. For each student, it propagates mastery using rules like:
```
If mastery(Linear Equations) → 0.5 * mastery(Quadratic Functions)
```

### Causal Layer Details
ACE is calculated using:
```
ACE(X → Y) = E[Y | do(X=1)] - E[Y | do(X=0)]
```
where X is a binary variable (e.g., "Time On Task" high/low).

### Agentic Grading Details
- Uses GPT-4o Mini's **vision capabilities (GPT-4o Mini is multimodal!)
- Images sent as base64 to OpenAI API
- Returns bounding boxes + feedback in structured JSON
- Gracefully falls back to sample data if API key is invalid
- Image is auto-cropped to fit container (800x600) with dark background

---

## Files & Folders
```
main el - causal/
├── api/                          # Vercel serverless functions
│   ├── __init__.py
│   ├── dashboard.py
│   ├── causal.py
│   └── analyze.py
├── agentic_ui/                    # Frontend
│   ├── index.html
│   ├── dashboard.html
│   ├── causal.html
│   ├── inference.html
│   ├── css/
│   │   ├── shared.css
│   │   └── style.css
│   └── js/
│       └── app.js
├── backend/                       # Local Flask backend
│   ├── app.py
│   └── requirements.txt
├── models/                        # Model code
│   ├── nscausalkt.py
│   ├── dkt.py
│   ├── sakt.py
│   └── akt.py
├── checkpoints/                   # Trained model & logs
│   ├── latest.pt
│   └── training_logs.txt
├── data/                          # Dataset
│   └── skill_builder_data.csv
├── train.py                       # Training script
├── eval.py                        # Evaluation script
├── inference.py                   # Inference script
├── vercel.json                    # Vercel configuration
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore
├── README.md                      # Deployment guide
├── RUN_LOCALLY.md                 # Local run guide
├── PROJECT_DOCUMENTATION.md       # This file!
├── .env                           # Environment variables (OpenAI key)
```
