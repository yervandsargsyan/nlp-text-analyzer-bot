from telegram import Update
from telegram.ext import ContextTypes
from models import get_sentiment

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Send me text to get sentiment")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    sentiment, score = await get_sentiment(text)
    await update.message.reply_text(f"score = {score:.2f}, result = {sentiment}")