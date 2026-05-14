# NS-CausalKT - Full Stack Deployment

## Overview
Complete integrated system with frontend, backend, and your trained NS-CausalKT model, ready for Vercel!

## Project Structure
```
your-repo/
├── api/                    # Vercel serverless functions
│   ├── __init__.py
│   ├── dashboard.py        # Dashboard metrics API
│   ├── causal.py           # Causal metrics API
│   └── analyze.py          # Inference API
├── agentic_ui/             # Frontend (4 pages)
│   ├── index.html          # Home
│   ├── dashboard.html      # Page 1 (model metrics)
│   ├── causal.html         # Page 2 (causal analysis)
│   ├── inference.html      # Page 3 (agentic grading)
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
├── models/                 # Your model code
├── checkpoints/            # Your trained model (50MB allowed!)
│   ├── latest.pt
│   └── training_logs.txt
├── data/                   # Dataset (50MB total allowed!)
├── vercel.json             # Vercel configuration
├── requirements.txt        # Python dependencies
└── .gitignore              # Git ignore (updated to allow models/data)
```

## Deploy to Vercel

### Step 1: Commit & Push to GitHub
```bash
git add .
git commit -m "Deploy NS-CausalKT"
git push
```

### Step 2: Import to Vercel
1. Go to [vercel.com/new](https://vercel.com/new)
2. Import your GitHub repository
3. Wait for Vercel to detect the configuration

### Step 3: Configure Environment Variables
In Vercel dashboard → Settings → Environment Variables:
- Add: `OPENAI_API_KEY` → Your OpenAI key (for inference page)

### Step 4: Deploy!
Click "Deploy"!

## Notes
- **Model & Dataset Committed**: Yes, up to 50MB total allowed on GitHub
- **Pages 1 & 2**: Use your real trained NS-CausalKT model
- **Page 3**: Uses GPT-4o Mini + your model insights
- **.gitignore updated**: Allows checkpoints/ and data/ folders
