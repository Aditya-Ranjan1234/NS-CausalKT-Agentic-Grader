# NS-CausalKT Integrated System

## Overview
Combined system with:
1. Dashboard - real model metrics from your trained NS-CausalKT
2. Causal Analysis - causal metrics and benchmarks
3. Inference - agentic grading with GPT-4o Mini and bounding boxes

## Folder Structure
```
agentic_ui/
├── index.html          # Home page
├── dashboard.html      # Page 1: Dashboard (uses real model metrics)
├── causal.html         # Page 2: Causal Analysis
├── inference.html      # Page 3: Inference/Agentic Grading
├── css/
│   └── style.css       # Combined styles
├── js/
│   └── app.js          # Inference UI logic
├── backend/
│   ├── app.py          # Flask backend (serves all pages + API)
│   └── requirements.txt
└── uploads/            # Temp upload folder
```

## Setup

1. **Install dependencies**:
   ```bash
   cd agentic_ui/backend
   pip install -r requirements.txt
   ```

2. **Set OpenAI API Key**:
   ```bash
   set OPENAI_API_KEY=your-api-key-here
   ```

3. **Run the backend**:
   ```bash
   python app.py
   ```

4. **Open the UI**:
   Open `agentic_ui/index.html` in your browser

## API Endpoints
- `GET /` - Home page
- `GET /dashboard.html` - Dashboard
- `GET /causal.html` - Causal Analysis
- `GET /inference.html` - Inference
- `GET /api/dashboard` - Get real model metrics (AUC, RMSE, ACC, etc.) from training logs
- `GET /api/causal` - Get causal metrics and benchmark data
- `POST /api/analyze` - Upload files for agentic grading

## Notes
- **Pages 1 & 2** use your trained NS-CausalKT model metrics from `checkpoints/training_logs.txt`
- **Page 3** uses GPT-4o Mini for agentic feedback and your NS-CausalKT for model insights
