import sys
import os
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from bot import start, handle_message

# Чтобы Python видел папку models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("Set BOT_TOKEN as an environment variable")

def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Обработчики
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Запуск бота
    app.run_polling()

if __name__ == "__main__":
    main()