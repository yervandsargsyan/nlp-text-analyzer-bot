# NLP Text Analyzer Bot
**NLP Text Analyzer Bot** is a Telegram bot that analyzes text using non-classical NLP ensemble models. It can identify content in messages and classify it according to a variety of criteria. Use case: automatically send messages from your profile to the bot and automatically delete unwanted comments using a filter.

## Features v0.8
- Multilingual sentiment analysis using ensemble models

## How to Run Locally
1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Set your Telegram bot token as an environment variable:
    **Linux / Mac**
    ```bash
    export BOT_TOKEN=your_telegram_bot_token
    ```
    **Windows**
    ```bat
    set BOT_TOKEN=your_telegram_bot_token
    ```
3. Run the bot:
    ```bash
    python bot/main.py
    ```

## Upcoming Updates
- Toxicity detection
- Emotional tone recognition
- Cross-platform automatic message analysis with optional deletion/blocking
