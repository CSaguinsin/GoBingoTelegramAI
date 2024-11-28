class TelegramView:
    @staticmethod
    async def send_welcome_message(update):
        await update.message.reply_text(
            "Welcome to GoBingo Telegram AI Bot! 👋\n\n"
            "Please upload your Identity Card photo."
        )

    @staticmethod
    async def send_processing_message(update, doc_type):
        await update.message.reply_text(f"Processing your {doc_type}...")

    @staticmethod
    async def send_error_message(update, error_msg):
        await update.message.reply_text(f"Sorry, there was an error: {error_msg}")

    @staticmethod
    async def send_validation_error(update, message):
        await update.message.reply_text(f"Image validation failed: {message}")

    @staticmethod
    async def send_extracted_text(update, doc_type, text):
        await update.message.reply_text(f"Extracted text from {doc_type}:\n\n{text}")

    @staticmethod
    async def request_next_document(update, doc_type):
        await update.message.reply_text(f"Please upload your {doc_type} photo.")

    @staticmethod
    async def send_completion_message(update):
        await update.message.reply_text("Thank you for using GoBingo Telegram AI bot! 🎉")

    @staticmethod
    async def send_cancel_message(update):
        await update.message.reply_text("Process cancelled. Send /start to begin again.")
