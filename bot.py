import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
from telegram.error import TimedOut, NetworkError
from dotenv import load_dotenv
from model import extract_text_from_image, validate_image
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger = logging.getLogger(__name__)

# Define conversation states
UPLOAD_ID, UPLOAD_LICENSE, UPLOAD_LOG = range(3)

# Maximum retries for file download
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

async def download_with_retry(photo, user_id, retry_count=0):
    try:
        photo_file = await photo.get_file()
        image_path = f"temp_id_{user_id}.jpg"
        await photo_file.download_to_drive(image_path)
        return image_path
    except (TimedOut, NetworkError) as e:
        if retry_count < MAX_RETRIES:
            logger.info(f"Retry {retry_count + 1}/{MAX_RETRIES} after error: {str(e)}")
            time.sleep(RETRY_DELAY)
            return await download_with_retry(photo, user_id, retry_count + 1)
        raise

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        "Welcome to GoBingo Telegram AI Bot! 👋\n\n"
        "Please upload your Identity Card photo."
    )
    return UPLOAD_ID

async def handle_id_card(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.message.from_user
    photo = update.message.photo[-1]
    
    await update.message.reply_text("Processing your Identity Card...")
    
    try:
        # Download photo with retry mechanism
        image_path = await download_with_retry(photo, user.id)
        
        # Validate image
        is_valid, message = validate_image(image_path)
        if not is_valid:
            logger.error(f"Image validation failed: {message}")
            await update.message.reply_text(f"Image validation failed: {message}")
            return UPLOAD_ID
        
        # Extract text using SmolVLM
        extracted_text = extract_text_from_image(image_path)
        await update.message.reply_text(f"Extracted text from ID Card:\n\n{extracted_text}")
        await update.message.reply_text("Please upload your Driver's License photo.")
        
        # Cleanup
        if os.path.exists(image_path):
            os.remove(image_path)
            
        return UPLOAD_LICENSE
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        await update.message.reply_text(
            "Sorry, there was an error processing your image. "
            "Please try uploading the ID Card photo again."
        )
        return UPLOAD_ID

async def handle_drivers_license(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.message.from_user
    photo = update.message.photo[-1]
    
    await update.message.reply_text("Processing your Driver's License...")
    
    try:
        # Download photo with retry mechanism
        image_path = await download_with_retry(photo, user.id)
        
        # Validate image
        is_valid, message = validate_image(image_path)
        if not is_valid:
            logger.error(f"Image validation failed: {message}")
            await update.message.reply_text(f"Image validation failed: {message}")
            return UPLOAD_LICENSE
        
        # Extract text using SmolVLM
        extracted_text = extract_text_from_image(image_path)
        await update.message.reply_text(f"Extracted text from Driver's License:\n\n{extracted_text}")
        await update.message.reply_text("Please upload your Log Card photo.")
        
        # Cleanup
        if os.path.exists(image_path):
            os.remove(image_path)
            
        return UPLOAD_LOG
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        await update.message.reply_text(
            "Sorry, there was an error processing your image. "
            "Please try uploading the Driver's License photo again."
        )
        return UPLOAD_LICENSE

async def handle_log_card(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.message.from_user
    photo = update.message.photo[-1]
    
    await update.message.reply_text("Processing your Log Card...")
    
    try:
        # Download photo with retry mechanism
        image_path = await download_with_retry(photo, user.id)
        
        # Validate image
        is_valid, message = validate_image(image_path)
        if not is_valid:
            logger.error(f"Image validation failed: {message}")
            await update.message.reply_text(f"Image validation failed: {message}")
            return UPLOAD_LOG
        
        # Extract text using SmolVLM
        extracted_text = extract_text_from_image(image_path)
        await update.message.reply_text(f"Extracted text from Log Card:\n\n{extracted_text}")
        await update.message.reply_text("Thank you for using GoBingo Telegram AI bot! 🎉")
        
        # Cleanup
        if os.path.exists(image_path):
            os.remove(image_path)
            
        return ConversationHandler.END
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        await update.message.reply_text(
            "Sorry, there was an error processing your image. "
            "Please try uploading the Log Card photo again."
        )
        return UPLOAD_LOG

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Process cancelled. Send /start to begin again.")
    return ConversationHandler.END

def main():
    # Initialize bot
    application = Application.builder().token(os.getenv('TELEGRAM_BOT_API')).build()

    # Create conversation handler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            UPLOAD_ID: [MessageHandler(filters.PHOTO, handle_id_card)],
            UPLOAD_LICENSE: [MessageHandler(filters.PHOTO, handle_drivers_license)],
            UPLOAD_LOG: [MessageHandler(filters.PHOTO, handle_log_card)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )

    # Add conversation handler
    application.add_handler(conv_handler)

    # Start the bot
    print("Bot started successfully!")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main() 