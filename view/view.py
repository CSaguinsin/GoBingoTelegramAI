import asyncio
from telegram import Update
from telegram.ext import CallbackContext
from telegram.error import TimedOut
import logging

logger = logging.getLogger(__name__)

class TelegramView:
    @staticmethod
    async def send_message(update: Update, text: str):
        """
        Sends a generic text message to the user.
        """
        await update.message.reply_text(text)

    @staticmethod
    async def send_welcome_message(update):
        await update.message.reply_text(
            "Welcome to GoBingo Telegram AI Bot! 👋\n\n"
            "Please upload your Identity Card photo."
        )


    @staticmethod
    async def send_model_loading_message(update):
        await update.message.reply_text(
            "Your data is secure. Our internal AI model extracts the necessary information directly from your uploaded documents in our AI Assistant, ensuring your personal information remains confidential and is not shared with any third parties. Thank you for trusting us.")

    @staticmethod
    async def send_processing_message(update, doc_type):
        try:
            await update.message.reply_text(
                f"Processing your {doc_type}...\n"
                "This typically takes 30-60 seconds... ⏳"
            )
        except Exception as e:
            logger.error(f"Failed to send processing message: {str(e)}")

    @staticmethod
    async def send_error_message(update, error_msg):
        await update.message.reply_text(f"Sorry, there was an error: {error_msg}")

    @staticmethod
    async def send_validation_error(update, message):
        await update.message.reply_text(f"Image validation failed: {message}")

    @staticmethod
    async def send_processing_complete(update, doc_type):
        await update.message.reply_text(f"{doc_type} processing completed successfully.")

    @staticmethod
    async def request_next_document(update, doc_type):
        await update.message.reply_text(f"Please upload your {doc_type} photo.")

    @staticmethod
    async def send_completion_message(update):
        await update.message.reply_text("Thank you for using GoBingo AI Assistant! 🎉")

    @staticmethod
    async def send_cancel_message(update):
        await update.message.reply_text("Process cancelled. Send /start to begin again.")

    @staticmethod
    async def send_data_saved_message(update):
        await update.message.reply_text(
            "All documents processed successfully! Data has been saved to the system. ✅"
        )

    @staticmethod
    async def send_data_save_error_message(update):
        await update.message.reply_text(
            "⚠️ Documents processed but there was an error saving the data. Please try again or contact support."
        )

    @staticmethod
    async def send_extracted_text(update, doc_type, text):
        """Send extracted text to user with formatting."""
        message = f"📄 Extracted information from {doc_type}:\n\n{text}"
        await update.message.reply_text(message)

    async def send_message_with_retry(self, update, text, max_retries=3, retry_delay=2):
        """Send message with retry mechanism for timeout handling"""
        for attempt in range(max_retries):
            try:
                return await update.message.reply_text(text)
            except TimedOut:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(retry_delay)
            except Exception as e:
                logger.error(f"Error sending message: {str(e)}")
                raise
