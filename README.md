# NLP Text Analyzer Bot

**NLP Text Analyzer Bot** is a Telegram bot that analyzes text using non-classical NLP ansamble models.  
It can identify content in messages and classify it according to a variety of criteria.
Use case: automatically send messages from your profile to the bot and automatically delete unwanted comments using a filter.

## Features v0.8

* multilingual context analysis (Sentiment Analysis) 

## How to Run Locally

1. Install dependencies:
   pip install -r requirements.txt
2. Set your Telegram bot token as an environment variable:
   export BOT\_TOKEN=your\_telegram\_bot\_token / set BOT\_TOKEN=your\_telegram\_bot\_token
3. Run the bot:
   python bot/main.py

## Upcoming updates
* Toxicity Detection
* Emotionak tone Recognition
* the ability to perform cross-platform automatic message analysis followed by deletion/blocking 
