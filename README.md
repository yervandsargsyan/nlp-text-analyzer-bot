# NLP Text Analyzer Bot

**NLP Text Analyzer Bot** is a Telegram bot that analyzes text using NLP models.  
It can detect sentiment, emotions, and toxic content in messages.

## Features

* Sentiment Analysis (positive/neutral/negative)
* Toxicity Detection
* Emotion Recognition (joy, anger, sadness, etc.)

## Project Structure

nlp-text-analyzer-bot/
├── bot/ # Telegram bot code
├── models/ # Model loader
├── LICENSE.txt # License file
├── requirements.txt # Python dependencies
└── README.md # Project description

## How to Run Locally

1. Install dependencies:
   pip install -r requirements.txt
2. Set your Telegram bot token as an environment variable:
   export BOT\_TOKEN=your\_telegram\_bot\_token / set BOT\_TOKEN=your\_telegram\_bot\_token
3. Run the bot:
   python bot/main.py
