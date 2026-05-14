# NS-CausalKT: Neural-Symbolic Causal Knowledge Tracing & Agentic Grading

[![License: MIT](https://img.shields.io/badge/License-MIT-cyan.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch: 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**NS-CausalKT** is a state-of-the-art Neural-Symbolic framework designed to transform AI-driven education from simple "score prediction" to deep "causal diagnostics." By combining the predictive power of Transformers with the transparency of Causal Inference (Pearl's Do-Calculus) and Symbolic Logic, the system identifies the root causes of student misconceptions.

---

## 🌟 Key Innovations

### 1. Neural-Symbolic Architecture
The model utilizes a **Hybrid Transformer-Logic** approach. While the Neural layer (Transformers) captures temporal patterns in learning, the Symbolic layer enforces mathematical constraints and logical prerequisites, ensuring the model's knowledge representation is educationally sound.

### 2. Causal Discovery (SCM)
Unlike traditional Knowledge Tracing (DKT, AKT) which only finds correlations, NS-CausalKT builds a **Structural Causal Model (DAG)**. It distinguishes between *effort-based success* (Time Spent) and *knowledge-based success* (Prior Mastery), allowing for "What-If" counterfactual simulations.

### 3. Agentic Multimodal Grading
The **Agentic System** integrates GPT-4o Mini with the NS-CausalKT backend. It can:
- **See**: Analyze handwritten answer sheets via Multimodal Vision.
- **Reason**: Map visual errors (e.g., a sign error in Algebra) to specific nodes in the Concept Dependency Graph.
- **Act**: Generate strategic feedback tailored to the student's causal mastery profile.

---

## 🚀 Research Metrics (Trained on ASSISTments 2009)

| Model Architecture | AUC (BaseLine) | AUC (Causal) | RMSE Delta | Efficiency Gain |
|--------------------|----------------|---------------|------------|-----------------|
| **NS-CausalKT-v2** | **0.824**      | **0.891**     | **-0.062** | **+6.1%**       |
| AKT (Baseline)     | 0.791          | 0.791         | +0.000     | Baseline        |
| DKT (Baseline)     | 0.805          | 0.812         | +0.002     | +1.2%           |

---

## 🛠️ Tech Stack

- **Core**: Python 3.10, PyTorch, NumPy, Pandas.
- **Causal Engine**: Pearl's Do-Calculus, Structural Causal Models (SCM).
- **Agentic Layer**: OpenAI GPT-4o Mini (Vision API).
- **Dashboard**: Vanilla JS, CSS3 (Modern Blue/Cyan Glassmorphism UI), SVG-based Dynamic Graphing.

---

## 📂 Project Structure

```text
NS-CausalKT-Agentic-Grader/
├── agentic_ui/            # Full-stack Research Dashboard
│   ├── causal.html        # Causal Analytics & Counterfactual Simulator
│   ├── inference.html     # Agentic Grading UI (Full-width marksheet analysis)
│   ├── dashboard.html     # Real-time Benchmark Monitoring
│   ├── js/app.js          # SVG Arrow Mapping & Graph Rendering Logic
│   └── css/style.css      # Professional Cyan-Slate Design System
├── models/                # Core Neural-Symbolic Causal Logic
├── checkpoints/           # Trained Model Weights & Benchmark Logs
├── data/                  # Preprocessed Math Datasets
├── docs/                  # Technical Documentation & Explanations
└── backend/               # Flask API for Model Serving
```

---

## 🔧 Installation & Deployment

### 1. Local Setup
```bash
# Clone the repository
git clone https://github.com/Aditya-Ranjan1234/NS-CausalKT-Agentic-Grader.git
cd NS-CausalKT-Agentic-Grader

# Install dependencies
pip install -r requirements.txt

# Run the backend
python agentic_ui/backend/app.py
```

### 2. Deployment (Vercel)
The project is configured for Vercel deployment out-of-the-box. Ensure your `OPENAI_API_KEY` is set in the environment variables for the Agentic Grading module.

---

## 📊 Visualization Key

- **ACE (Average Causal Effect)**: Measures the intensity of the link between concepts.
- **CDG (Concept Dependency Graph)**: A symbolic map of math topic prerequisites.
- **SCM DAG**: A directed acyclic graph representing the global causal factors of learning.

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*“Moving beyond correlation to causal understanding in AI-driven education.”*
