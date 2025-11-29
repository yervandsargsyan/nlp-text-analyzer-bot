# NLP Text Analyzer Bot

**NLP Text Analyzer Bot** is a Telegram bot that analyzes text using non-classical NLP ensemble models.
It can identify content in messages and classify it according to a variety of criteria.
Use case: automatically send messages from your profile to the bot and automatically delete unwanted comments using a filter.

## Features v0.8

* Multilingual sentiment analysis using ensemble models

## How to Run Locally
1. Install dependencies:<br>
```pip install -r requirements.txt```<br>
2. Set your Telegram bot token as an environment variable:<br>
   Linux/Mac: ```bash export BOT\_TOKEN=your\_telegram\_bot\_token ```<br>
   Windows: ```bash set BOT\_TOKEN=your\_telegram\_bot\_token ```<br>
3. Run the bot:<br>
   python ```bash bot/main.py```

## Upcoming updates
* Toxicity detection
* Emotional tone recognition
* Cross-platform automatic message analysis with optional deletion/blocking
