#!/bin/bash
# Setup script for Kisan Telegram Bot

set -e

echo "🌾 Kisan Telegram Bot - Setup Script"
echo "======================================"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo "✓ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1

# Install requirements
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create .env if not exists
if [ ! -f ".env" ]; then
    echo "📋 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your TELEGRAM_BOT_TOKEN"
else
    echo "✓ .env file already exists"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models
mkdir -p data
mkdir -p logs
mkdir -p /tmp/kisan_images

# Initialize database
echo "🗄️  Initializing database..."
python -c "from app.database.database import init_db; init_db(); print('✓ Database initialized')"

# Check if diseases.csv exists and import
if [ -f "data/diseases.csv" ]; then
    echo "📥 Importing disease database..."
    python scripts/import_diseases.py data/diseases.csv || echo "⚠️  Warning: Could not import diseases (may need manual setup)"
else
    echo "⚠️  data/diseases.csv not found - skipping disease import"
fi

echo ""
echo "======================================"
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your TELEGRAM_BOT_TOKEN"
echo "2. Run: python -m uvicorn app.main:app --reload"
echo "3. Open Telegram and start the bot"
echo ""
echo "For more details, see README.md"
