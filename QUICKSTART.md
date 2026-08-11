# ⚡ Quick Start Guide

Get the Kisan Telegram Bot running in **5 minutes**.

## Prerequisites

- Python 3.11+
- A Telegram account
- A terminal/command prompt
- ~500MB disk space

## Step 1: Get Bot Token (2 min)

1. Open Telegram and search for `@BotFather`
2. Send `/newbot`
3. Answer:
   - **Name:** Kisan Crop Assistant
   - **Username:** kisan_crop_bot_YOURNAME (must be unique)
4. Copy the token provided (looks like: `123456:ABCDefghIjklMnoPqRsTuvWxYz`)

## Step 2: Clone & Setup (2 min)

```bash
# Clone repository
git clone https://github.com/yourusername/kisan-telegram-bot.git
cd kisan-telegram-bot

# Run setup script
bash setup.sh

# Or manual setup:
# python3 -m venv venv
# source venv/bin/activate  (or venv\Scripts\activate on Windows)
# pip install -r requirements.txt
```

## Step 3: Configure Bot (1 min)

Edit `.env` file:

```env
TELEGRAM_BOT_TOKEN=your_token_here_123456:ABCDefghIjklMnoPqRsTuvWxYz
DATABASE_URL=sqlite:///./kisan_bot.db
USE_WEBHOOK=false
ENVIRONMENT=development
```

## Step 4: Run Bot

```bash
# Activate virtual environment (if not already)
source venv/bin/activate

# Start server
python -m uvicorn app.main:app --reload
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

## Step 5: Test Bot

1. **Open Telegram**
2. **Search for:** @kisan_crop_bot_YOURNAME (the username you chose)
3. **Send:** `/start`

Bot should reply with welcome message:
```
🌱 Welcome to Kisan Crop Assistant!

I can help you identify crop diseases...
```

## Try Commands

```
/help           → Show help message
/language       → Change language (English/Hindi/Hinglish)
/about          → About this bot

[Send photo]    → Bot analyzes crop disease
```

## Troubleshooting

### Bot not responding?

**Check if server is running:**
```bash
curl http://localhost:8000/health
```

Should return:
```json
{"status": "healthy", "service": "kisan-telegram-bot", ...}
```

**Check bot token:**
```bash
# Make sure .env has correct token
cat .env | grep TELEGRAM_BOT_TOKEN
```

**Check logs:**
The terminal running `uvicorn` shows live logs. Look for errors.

### Model not working?

The bot works without a trained ML model - it will show:
```
⚠️ ML model not configured
```

To use image detection:
1. Train a model: `python training/train_model.py`
2. Place files in `models/` directory
3. Restart bot

### Database errors?

```bash
# Reinitialize database
python -c "from app.database.database import init_db; init_db()"
```

## What Happens Next?

After `/start`:
1. **Send a crop photo** → Bot analyzes it
2. **Ask questions** → Bot uses rule-based NLP
3. **Get responses** → From database or templates

## Example Interaction

**You:** Send photo of tomato with brown spots  
**Bot:** 
```
🌱 Crop: Tomato
⚠️ Possible Disease: Early Blight
📊 Confidence: 92%

🔍 Symptoms:
Dark circular spots on leaves...

🛠 Management:
• Remove infected leaves
• Improve air circulation
...
```

**You:** "क्या करें?" (What to do?)  
**Bot:** Returns management info in Hindi (remembers context)

## Next: Deploy to Render

Once working locally, deploy to production:

See **[DEPLOYMENT.md](./DEPLOYMENT.md)** or **[README.md → Deployment (Render)](./README.md#-deployment-render)**

## Need Help?

- **Check logs** in terminal running uvicorn
- **Read README.md** for detailed documentation
- **Review test files** in `tests/` directory

## File Structure (What Gets Created)

```
✅ kisan-telegram-bot/
   ✅ app/                  (Application code)
   ✅ models/               (ML models - empty for now)
   ✅ data/                 (Disease database - CSV)
   ✅ scripts/              (Utility scripts)
   ✅ training/             (Model training code)
   ✅ kisan_bot.db          (SQLite database - created)
   ✅ .env                  (Your config - NEVER commit!)
```

---

**Total time: ~5 minutes**  
**Ready to use: ✅ YES**

Enjoy! 🌾
