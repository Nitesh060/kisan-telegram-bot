#!/usr/bin/env python3
"""
Setup Telegram webhook for production deployment
"""

import sys
import os
import requests
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings


def get_webhook_info(token: str):
    """Get current webhook info."""
    url = f"https://api.telegram.org/bot{token}/getWebhookInfo"
    try:
        response = requests.post(url)
        return response.json()
    except Exception as e:
        print(f"❌ Error getting webhook info: {e}")
        return None


def set_webhook(token: str, webhook_url: str):
    """Set webhook URL."""
    url = f"https://api.telegram.org/bot{token}/setWebhook"
    
    data = {
        "url": webhook_url,
        "drop_pending_updates": True,
    }
    
    try:
        response = requests.post(url, json=data)
        result = response.json()
        
        if result.get("ok"):
            print(f"✅ Webhook set successfully!")
            print(f"   URL: {webhook_url}")
            return True
        else:
            print(f"❌ Error: {result.get('description')}")
            return False
    except Exception as e:
        print(f"❌ Error setting webhook: {e}")
        return False


def delete_webhook(token: str):
    """Delete webhook (switch to polling)."""
    url = f"https://api.telegram.org/bot{token}/deleteWebhook"
    
    try:
        response = requests.post(url)
        result = response.json()
        
        if result.get("ok"):
            print("✅ Webhook deleted - switching to polling mode")
            return True
        else:
            print(f"❌ Error: {result.get('description')}")
            return False
    except Exception as e:
        print(f"❌ Error deleting webhook: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Configure Telegram webhook")
    parser.add_argument(
        "action",
        choices=["set", "delete", "check"],
        help="Action to perform"
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Telegram bot token (defaults to TELEGRAM_BOT_TOKEN env var)"
    )
    parser.add_argument(
        "--url",
        type=str,
        help="Webhook URL (defaults to WEBHOOK_URL env var)"
    )
    
    args = parser.parse_args()
    
    # Get token
    token = args.token or settings.TELEGRAM_BOT_TOKEN
    if not token:
        print("❌ Telegram bot token not found. Set TELEGRAM_BOT_TOKEN env var or use --token")
        sys.exit(1)
    
    print("🔧 Telegram Webhook Configuration")
    print("="*50)
    print(f"Token: {token[:10]}...")
    print()
    
    if args.action == "check":
        print("📡 Checking current webhook info...")
        info = get_webhook_info(token)
        if info and info.get("ok"):
            webhook = info.get("result", {})
            print(f"✅ Webhook URL: {webhook.get('url', 'Not set')}")
            print(f"   Pending updates: {webhook.get('pending_update_count', 0)}")
        else:
            print("❌ Could not retrieve webhook info")
    
    elif args.action == "set":
        url = args.url or settings.WEBHOOK_URL
        if not url:
            print("❌ Webhook URL not found. Set WEBHOOK_URL env var or use --url")
            sys.exit(1)
        
        print(f"📡 Setting webhook to: {url}")
        if set_webhook(token, url):
            sys.exit(0)
        else:
            sys.exit(1)
    
    elif args.action == "delete":
        print("🗑️  Deleting webhook...")
        if delete_webhook(token):
            sys.exit(0)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
