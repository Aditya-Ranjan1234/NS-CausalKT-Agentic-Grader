# NS-CausalKT - Vercel Deployment

## Deploy to Vercel

### 1. Push to GitHub
First, push your entire project to GitHub.

### 2. Import to Vercel
1. Go to [vercel.com](https://vercel.com)
2. Click "New Project"
3. Import your GitHub repository

### 3. Configure Environment Variables
In Vercel dashboard:
- Add `OPENAI_API_KEY` with your OpenAI key (for the inference page)

### 4. Deploy!
Click "Deploy" - Vercel will automatically detect the configuration.

## Project Structure for Vercel
```
your-repo/
├── api/                    # Vercel serverless functions
│   ├── dashboard.py
│   ├── causal.py
│   └── analyze.py
├── agentic_ui/             # Static files
│   ├── index.html
│   ├── dashboard.html
│   ├── causal.html
│   ├── inference.html
│   ├── css/
│   └── js/
├── checkpoints/            # Training logs (NO .pt files!)
│   └── training_logs.txt
├── vercel.json             # Vercel config
└── .gitignore
```

## Notes
- **DO NOT commit .pt model files** or large datasets to GitHub - they're too big!
- The .gitignore excludes model checkpoints except for the log files
- Dashboard uses your training_logs.txt for metrics
