# Run NS-CausalKT Locally

## Option 1: Use Flask Backend (Full Stack)

### Step 1: Navigate to backend
```bash
cd agentic_ui/backend
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Set OpenAI API Key (optional for inference page)
```bash
set OPENAI_API_KEY=your-api-key-here
```

### Step 4: Run the backend
```bash
python app.py
```

### Step 5: Open the UI
Open `agentic_ui/index.html` in your browser!

---

## Option 2: Use Vercel Dev (for testing)

### Step 1: Install Vercel CLI
```bash
npm install -g vercel
```

### Step 2: Login to Vercel
```bash
vercel login
```

### Step 3: Run locally
```bash
vercel dev
```

### Step 4: Open the local URL
Vercel will give you a local URL (like http://localhost:3000)

---

## Notes
- Your trained model is in `checkpoints/latest.pt`
- Dataset is in `data/`
- Training logs are in `checkpoints/training_logs.txt`
