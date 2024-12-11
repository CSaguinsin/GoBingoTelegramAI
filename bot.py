import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ConversationHandler, Defaults
from dotenv import load_dotenv
from controller.controller import TelegramController, UPLOAD_ID, UPLOAD_LICENSE, UPLOAD_LOG, STATE_REFERRER_NAME, STATE_REFERRER_CONTACT, STATE_REFERRER_DEALERSHIP

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

def main() -> None:
    # Initialize application with default timeouts
    application = (
        Application.builder()
        .token(os.getenv('TELEGRAM_BOT_API'))
        .read_timeout(int(os.getenv('TELEGRAM_READ_TIMEOUT', 30)))
        .write_timeout(int(os.getenv('TELEGRAM_WRITE_TIMEOUT', 30)))
        .connect_timeout(int(os.getenv('TELEGRAM_CONNECT_TIMEOUT', 20)))
        .pool_timeout(int(os.getenv('TELEGRAM_TIMEOUT', 30)))
        .build()
    )
    
    controller = TelegramController()

    # Create conversation handler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', controller.start)],
        states={
            UPLOAD_ID: [MessageHandler(filters.PHOTO, controller.handle_id_card)],
            UPLOAD_LICENSE: [MessageHandler(filters.PHOTO, controller.handle_drivers_license)],
            UPLOAD_LOG: [MessageHandler(filters.PHOTO, controller.handle_log_card)],
            STATE_REFERRER_NAME: [MessageHandler(filters.TEXT, controller.collect_referrer_name)],
            STATE_REFERRER_CONTACT: [MessageHandler(filters.TEXT, controller.collect_referrer_contact)],
            STATE_REFERRER_DEALERSHIP: [MessageHandler(filters.TEXT, controller.collect_referrer_dealership)],
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