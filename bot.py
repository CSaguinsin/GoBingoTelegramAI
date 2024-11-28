import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ConversationHandler
from dotenv import load_dotenv
from controller.controller import TelegramController, UPLOAD_ID, UPLOAD_LICENSE, UPLOAD_LOG

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

def main() -> None:
    # Initialize bot and controller
    application = Application.builder().token(os.getenv('TELEGRAM_BOT_API')).build()
    controller = TelegramController()

    # Create conversation handler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', controller.start)],
        states={
            UPLOAD_ID: [MessageHandler(filters.PHOTO, controller.handle_id_card)],
            UPLOAD_LICENSE: [MessageHandler(filters.PHOTO, controller.handle_drivers_license)],
            UPLOAD_LOG: [MessageHandler(filters.PHOTO, controller.handle_log_card)],
        },
        fallbacks=[CommandHandler('cancel', controller.cancel)],
    )

    # Add conversation handler
    application.add_handler(conv_handler)

    # Start the bot
    logger.info("Bot started successfully!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main() 