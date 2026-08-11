# 🌾 Kisan Telegram Crop Disease Detection Bot

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A **production-ready Telegram bot** that helps farmers identify crop diseases using **computer vision** and **machine learning** — without any LLM or generative AI dependencies.

## 🎯 Features

✅ **Image-based Disease Detection** — Farmers send crop/leaf photos, bot identifies disease using transfer learning (MobileNetV3)  
✅ **Verified Disease Database** — PostgreSQL backend with structured disease information (symptoms, management, prevention)  
✅ **Rule-based Intent Detection** — Understand farmer questions (management, symptoms, prevention) without NLP libraries  
✅ **Multi-language Support** — English, Hindi, and Hinglish responses  
✅ **Confidence Thresholds** — Reliable predictions with confidence levels  
✅ **Session Context Awareness** — Bot remembers last identified disease for follow-up questions  
✅ **Production Ready** — FastAPI, async handlers, error handling, logging  
✅ **No LLM Costs** — Zero OpenAI, Gemini, or Claude API dependencies  
✅ **Render/Cloud Ready** — Docker, webhook support, environment-based config  

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [Prerequisites](#prerequisites)
- [Quick Start (Local Development)](#quick-start-local-development)
- [Database Setup](#database-setup)
- [ML Model Setup](#ml-model-setup)
- [Telegram Bot Setup](#telegram-bot-setup)
- [Running Locally](#running-locally)
- [Deployment (Render)](#deployment-render)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    TELEGRAM FARMER                           │
│                (sends image + question)                      │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │    FASTAPI + UVICORN SERVER     │
        │  (Webhook or Polling Mode)      │
        └────────────────┬────────────────┘
                         │
        ┌────────────────┴─────────────────────────┐
        │                                          │
   ┌────▼─────┐  ┌──────────────┐  ┌────────────┐
   │  ML MODEL │  │ RULE-BASED   │  │ POSTGRESQL │
   │(MobileNet │  │ NLP (Intent) │  │ DATABASE   │
   │   V3)     │  │              │  │(Diseases)  │
   └────┬──────┘  └──────────────┘  └────────────┘
        │
        └────────────────┬──────────────────────┐
                         │                      │
                  ┌──────▼──────┐         ┌────▼────┐
                  │ CONFIDENCE  │         │ RESPONSE│
                  │ VALIDATION  │         │GENERATOR│
                  └─────────────┘         └─────────┘
                         │
                  ┌──────▼──────┐
                  │   TELEGRAM  │
                  │   API       │
                  └─────────────┘
                         │
                  ┌──────▼──────┐
                  │   FARMER    │
                  │   (Reply)   │
                  └─────────────┘
```

---

## 📦 Prerequisites

### System Requirements
- **Python 3.11+**
- **PostgreSQL 12+** (or SQLite for local dev)
- **Git**
- **pip** (Python package manager)

### Software
- Telegram (mobile app or web.telegram.org)
- A code editor (VS Code, PyCharm, etc.)
- A way to run Python scripts

---

## 🚀 Quick Start (Local Development)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/kisan-telegram-bot.git
cd kisan-telegram-bot
```

### Step 2: Create Virtual Environment

```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

```bash
cp .env.example .env
```

Edit `.env`:
```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
DATABASE_URL=sqlite:///./kisan_bot.db
USE_WEBHOOK=false
ENVIRONMENT=development
```

### Step 5: Initialize Database

```bash
python -c "from app.database.database import init_db; init_db()"
```

### Step 6: Import Disease Database

```bash
python scripts/import_diseases.py data/diseases.csv
```

### Step 7: Run Bot (Polling Mode)

```bash
python -m uvicorn app.main:app --reload
```

Bot will start on `http://localhost:8000`

---

## 🗄️ Database Setup

### Local Development (SQLite)

Already configured in `.env.example`:
```env
DATABASE_URL=sqlite:///./kisan_bot.db
```

Just run:
```bash
python -c "from app.database.database import init_db; init_db()"
```

### Production (PostgreSQL)

#### Option A: Local PostgreSQL

1. **Install PostgreSQL**
   ```bash
   # macOS (Homebrew)
   brew install postgresql
   brew services start postgresql
   
   # Ubuntu
   sudo apt-get install postgresql postgresql-contrib
   
   # Windows: Download from https://www.postgresql.org/download/windows/
   ```

2. **Create Database**
   ```bash
   psql -U postgres
   
   CREATE DATABASE kisan_bot;
   CREATE USER kisan_user WITH PASSWORD 'secure_password';
   GRANT ALL PRIVILEGES ON DATABASE kisan_bot TO kisan_user;
   ```

3. **Update `.env`**
   ```env
   DATABASE_URL=postgresql+psycopg2://kisan_user:secure_password@localhost:5432/kisan_bot
   ```

#### Option B: Render PostgreSQL (Automatic)

When deploying on Render, use the PostgreSQL add-on. The `DATABASE_URL` will be automatically provided.

### Import Disease Data

```bash
# Basic import
python scripts/import_diseases.py

# Custom CSV file
python scripts/import_diseases.py path/to/your/diseases.csv
```

**CSV Format (Required Columns):**
```csv
crop,disease_name,disease_type,symptoms,management,...
Tomato,Early Blight,Fungal,"Dark circular spots...",
```

---

## 🤖 ML Model Setup

### Option A: Use Pre-trained Model

If you have a trained model:

1. **Place Model Files**
   ```
   models/
   ├── crop_disease_model.keras
   └── class_names.json
   ```

2. **Class Names Format** (`models/class_names.json`):
   ```json
   {
     "0": "Tomato_Early_Blight",
     "1": "Tomato_Late_Blight",
     "2": "Rice_Blast",
     ...
   }
   ```

3. **Test Model**
   ```bash
   python -c "from app.ml.inference import ModelInference; m = ModelInference(); print(m.get_model_info())"
   ```

### Option B: Train Your Own Model

**Dataset Structure:**
```
data/
├── training/
│   ├── Tomato_Early_Blight/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── Tomato_Late_Blight/
│   └── ...
├── validation/
│   └── (same structure)
└── test/
    └── (same structure)
```

**Training:**
```bash
python training/train_model.py \
  --train-dir data/training \
  --val-dir data/validation \
  --test-dir data/test \
  --epochs 50 \
  --batch-size 32 \
  --model-path models/crop_disease_model.keras
```

**Output:**
- `models/crop_disease_model.keras` — Trained model
- `models/class_names.json` — Class name mapping

---

## 🤖 Telegram Bot Setup

### Step 1: Create Bot with BotFather

1. **Open Telegram** and search for `@BotFather`
2. **Send** `/newbot`
3. **Answer questions:**
   - Bot name: `Kisan Crop Assistant`
   - Username: `kisan_crop_bot` (must be unique)
4. **BotFather replies** with your bot token:
   ```
   Use this token to access the HTTP API:
   123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
   ```

### Step 2: Save Bot Token

Add to `.env`:
```env
TELEGRAM_BOT_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11
```

### Step 3: Set Bot Commands (Optional)

In BotFather, send:
```
/mybots
(select your bot)
/setcommands
```

Paste:
```
start - Start the bot
help - Show help message
language - Change language
about - About this bot
```

---

## 🏃 Running Locally

### Terminal 1: Start FastAPI Server

```bash
python -m uvicorn app.main:app --reload --port 8000
```

Server starts on `http://localhost:8000`

### Terminal 2: (Optional) Start Polling Endpoint

```bash
# Already handled by FastAPI, but you can trigger polling via:
curl -X POST http://localhost:8000/telegram/polling
```

### Test Bot

1. **Open Telegram**
2. **Search for your bot** (username you set with BotFather)
3. **Send** `/start`
4. **Test commands:**
   - `/help` — Show help
   - `/language` — Change language
   - **Send photo** — Bot analyzes image

### Check Health

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "service": "kisan-telegram-bot",
  "version": "1.0.0"
}
```

---

## 🌐 Deployment (Render)

### Step 1: Push to GitHub

```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### Step 2: Create Render Account

1. **Sign up** at https://render.com (free tier available)
2. **Connect GitHub** account

### Step 3: Deploy Web Service

1. **Dashboard → New → Web Service**
2. **Select GitHub repository**
3. **Settings:**
   - **Environment:** Python 3.11
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** Free (or Starter+)

4. **Deploy** — Render builds and starts your bot

### Step 4: Add PostgreSQL Database

1. **Dashboard → New → PostgreSQL**
2. **Create** database
3. **Copy DATABASE_URL** from database settings
4. **Add to Web Service environment variables:**
   ```
   KEY: DATABASE_URL
   VALUE: (paste from PostgreSQL)
   ```

### Step 5: Set Environment Variables

Go to **Web Service → Environment:**

```env
TELEGRAM_BOT_TOKEN=your_bot_token
USE_WEBHOOK=true
WEBHOOK_URL=https://your-service-name.onrender.com/telegram/webhook
DATABASE_URL=(from PostgreSQL)
ENVIRONMENT=production
```

### Step 6: Configure Telegram Webhook

Once deployed, update webhook:

```bash
curl -X POST https://api.telegram.org/bot<YOUR_BOT_TOKEN>/setWebhook \
  -H "Content-Type: application/json" \
  -d '{"url": "https://your-service-name.onrender.com/telegram/webhook"}'
```

Or run locally:
```python
import requests
token = "YOUR_BOT_TOKEN"
url = f"https://api.telegram.org/bot{token}/setWebhook"
response = requests.post(url, json={"url": "https://your-service.onrender.com/telegram/webhook"})
print(response.json())
```

### Step 7: Initialize Database

SSH into Render service and run:
```bash
python -c "from app.database.database import init_db; init_db()"
python scripts/import_diseases.py data/diseases.csv
```

Or use Render Shell:
1. Go to **Web Service → Shell**
2. Run the commands above

---

## 📡 API Endpoints

### Health & Status

```bash
# Health check
GET /health
→ {"status": "healthy", "service": "kisan-telegram-bot", "version": "1.0.0"}

# Database health
GET /database/health
→ {"status": "connected", "database": "sqlite:///./kisan_bot.db"}

# ML model health
GET /ml/health
→ {"status": "ready", "model_path": "models/crop_disease_model.keras"}
```

### Telegram

```bash
# Webhook endpoint (POST)
POST /telegram/webhook
(Telegram sends updates here)

# Polling trigger (POST)
POST /telegram/polling
→ {"status": "polling started"}
```

---

## 📁 Project Structure

```
kisan-telegram-bot/
│
├── app/                          # Main application
│   ├── __init__.py
│   ├── main.py                   # FastAPI entry point
│   ├── config.py                 # Configuration management
│   │
│   ├── bot/                      # Telegram bot logic
│   │   ├── handlers.py           # Command/message handlers
│   │   ├── telegram_service.py   # Telegram API wrapper
│   │   └── keyboards.py          # Keyboard layouts (optional)
│   │
│   ├── database/                 # Database layer
│   │   ├── database.py           # SQLAlchemy setup
│   │   ├── models.py             # ORM models
│   │   └── repository.py         # Data access layer
│   │
│   ├── ml/                       # Machine learning
│   │   ├── inference.py          # Model loading & inference
│   │   └── preprocessing.py      # Image preprocessing
│   │
│   ├── nlp/                      # Natural language processing
│   │   ├── intent.py             # Intent detection (rule-based)
│   │   └── keywords.py           # Keyword mappings
│   │
│   ├── responses/                # Response generation
│   │   ├── templates.py          # Message templates
│   │   └── generator.py          # Template rendering
│   │
│   └── services/                 # Business logic
│       └── disease_service.py    # Disease operations
│
├── models/                       # ML models
│   ├── crop_disease_model.keras  # (place trained model here)
│   └── class_names.json          # Class name mapping
│
├── data/                         # Data files
│   └── diseases.csv              # Disease database (CSV)
│
├── training/                     # ML training code
│   └── train_model.py            # Transfer learning script
│
├── scripts/                      # Utility scripts
│   └── import_diseases.py        # CSV → DB importer
│
├── tests/                        # Unit tests
│   ├── test_intent.py
│   ├── test_disease_service.py
│   └── test_confidence.py
│
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Docker config
├── render.yaml                   # Render deployment config
├── .env.example                  # Environment template
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

---

## 🧪 Testing

### Run Unit Tests

```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_intent.py -v

# With coverage
pytest --cov=app tests/
```

### Manual Testing

**Test intent detection:**
```python
from app.nlp.intent import IntentDetector
intent = IntentDetector.detect("What disease is this?")
print(intent)  # Should print: DIAGNOSIS
```

**Test image preprocessing:**
```python
from app.ml.preprocessing import preprocess_image, validate_image

# Check image validity
is_valid, msg = validate_image("path/to/image.jpg")
print(f"Valid: {is_valid}, Message: {msg}")

# Preprocess
image_array = preprocess_image("path/to/image.jpg")
print(f"Shape: {image_array.shape}")  # Should be (1, 224, 224, 3)
```

**Test disease database:**
```python
from app.database.database import get_session
from app.database.repository import DiseaseRepository

session = get_session()
disease = DiseaseRepository.get_disease(session, "Tomato", "Early Blight")
print(disease.symptoms if disease else "Not found")
session.close()
```

---

## 🐛 Troubleshooting

### Bot Not Responding

**Check bot is running:**
```bash
curl http://localhost:8000/health
```

**Check Telegram bot token:**
```bash
python -c "from app.config import settings; print('Token:', settings.TELEGRAM_BOT_TOKEN[:10])"
```

**Check logs:**
```bash
# If running with: python -m uvicorn app.main:app
# Logs appear in terminal. For file logs:
python -m uvicorn app.main:app > logs.txt 2>&1
```

### ML Model Not Loading

**Check model files exist:**
```bash
ls models/crop_disease_model.keras
ls models/class_names.json
```

**Check model health:**
```bash
curl http://localhost:8000/ml/health
```

**Check TensorFlow installation:**
```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
```

### Database Connection Error

**Check database URL:**
```bash
python -c "from app.config import settings; print(settings.DATABASE_URL)"
```

**Test PostgreSQL connection:**
```bash
# Install psycopg2 if not already
pip install psycopg2-binary

python -c "import psycopg2; print('psycopg2 works')"
```

**Initialize database:**
```bash
python -c "from app.database.database import init_db; init_db()"
```

### Webhook Issues (Render)

**Verify webhook URL:**
```bash
python -c "from app.config import settings; print(f'Webhook URL: {settings.WEBHOOK_URL}')"
```

**Check webhook status:**
```bash
curl -X POST https://api.telegram.org/bot<TOKEN>/getWebhookInfo
```

**Reset webhook:**
```bash
python -c "
import requests
token = 'YOUR_TOKEN'
url = f'https://api.telegram.org/bot{token}/deleteWebhook'
r = requests.post(url)
print(r.json())
"
```

### Image Processing Errors

**Check Pillow installation:**
```bash
python -c "from PIL import Image; print('Pillow works')"
```

**Check image format:**
```bash
python -c "
from app.ml.preprocessing import validate_image
is_valid, msg = validate_image('path/to/image.jpg')
print(f'Valid: {is_valid}, Message: {msg}')
"
```

---

## 📝 Example Conversation

**Farmer:** Sends photo of tomato leaf with brown spots  
**Bot:** Analyzes image → Identifies as "Tomato Early Blight" (92% confidence)

```
🌱 Crop: Tomato
⚠️ Possible Disease: Early Blight
📊 Confidence: 92%

🔍 Symptoms:
Dark circular spots appear on leaves, starting from lower leaves...

🛠 Management:
• Remove severely infected leaves.
• Avoid overhead irrigation.
• Apply fungicide if needed.

🛡 Prevention:
• Crop rotation (2-3 years)
• Use disease-free seeds
• Proper spacing

⚠️ Important Note:
This is an AI-based preliminary identification...
```

**Farmer:** "इसका इलाज कया है?" (What's the treatment?)  
**Bot:** Remembers "Tomato Early Blight" from context → Returns management info in Hindi

---

## 🔐 Security Notes

⚠️ **NEVER commit `.env` file** with real tokens  
✅ Use `.env.example` as template  
✅ Store secrets in Render environment variables  
✅ Rotate bot token if accidentally exposed  
✅ Use strong PostgreSQL passwords  
✅ Enable HTTPS for webhooks (Render does this automatically)

---

## 📜 License

MIT License — See LICENSE file

---

## 🤝 Contributing

1. Fork repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Commit: `git commit -m "Add feature"`
4. Push: `git push origin feature/your-feature`
5. Submit pull request

---

## 📞 Support

For issues or questions:
- **GitHub Issues:** [Create an issue](https://github.com/yourusername/kisan-telegram-bot/issues)
- **Email:** your-email@example.com

---

## 🌾 Made for Farmers

This bot is designed to help Indian farmers (किसान) identify crop diseases and get management guidance in their preferred language.

**Built with ❤️ by Annapurna Finance (AFPL)**
