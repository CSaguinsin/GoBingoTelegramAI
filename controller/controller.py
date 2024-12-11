from telegram import Update, File
from telegram.ext import ContextTypes, ConversationHandler
from model import process_document, validate_image
from view.view import TelegramView
from telegram.error import TimedOut, NetworkError
import os
import logging
import asyncio
from typing import Tuple, Optional
from datetime import datetime
import shutil
from services.monday_service import MondayService

logger = logging.getLogger(__name__)

# Define conversation states
STATE_UPLOAD_ID = 0
STATE_UPLOAD_LICENSE = 1
STATE_UPLOAD_LOG = 2
MAX_RETRIES = 3

STATE_REFERRER_NAME = 3
STATE_REFERRER_CONTACT = 4
STATE_REFERRER_DEALERSHIP = 5

# Exported conversation states
UPLOAD_ID = STATE_UPLOAD_ID
UPLOAD_LICENSE = STATE_UPLOAD_LICENSE
UPLOAD_LOG = STATE_UPLOAD_LOG

# Exported symbols for import in other modules
__all__ = [
    "TelegramController",
    "UPLOAD_ID",
    "UPLOAD_LICENSE",
    "UPLOAD_LOG",
]

class TelegramController:
    def __init__(self) -> None:
        self.view = TelegramView()
        self.monday_service = MondayService()
        self.extracted_data = {}

    async def download_with_retry(self, photo, user_id: int, doc_type: str, max_retries: int = MAX_RETRIES) -> Tuple[str, str]:
        """
        Download photo with retry mechanism and save to both temporary and permanent locations.
        """
        for attempt in range(max_retries):
            try:
                temp_dir = os.path.join('temp_documents')
                perm_dir = os.path.join('image_documents', doc_type)
                os.makedirs(temp_dir, exist_ok=True)
                os.makedirs(perm_dir, exist_ok=True)

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{user_id}_{timestamp}.jpg"
                temp_path = os.path.join(temp_dir, filename)
                saved_path = os.path.join(perm_dir, filename)

                file = await photo.get_file()
                await file.download_to_drive(temp_path)
                shutil.copy2(temp_path, saved_path)

                logger.info(f"Successfully downloaded and saved image. Temp: {temp_path}, Saved: {saved_path}")
                return saved_path, temp_path
            except Exception as e:
                logger.error(f"Download attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
        raise ValueError("Failed to download image after maximum retries")

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        await self.view.send_welcome_message(update)
        return UPLOAD_ID

    async def handle_id_card(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        user = update.message.from_user
        photo = update.message.photo[-1]

        await self.view.send_processing_message(update, "ID Card")
        temp_path = None

        try:
            saved_path, temp_path = await self.download_with_retry(photo, user.id, 'id_card')
            if not saved_path or not temp_path:
                raise ValueError("Failed to download image")

            await self.view.send_model_loading_message(update)
            is_valid, message = validate_image(saved_path)
            if not is_valid:
                logger.error(f"Image validation failed: {message}")
                await self.view.send_validation_error(update, message)
                return UPLOAD_ID

            extracted_text = await asyncio.wait_for(
                asyncio.to_thread(process_document, saved_path, 'id_card'),
                timeout=300.0
            )

            if extracted_text and "No data found" not in extracted_text:
                data_dict = {
                    line.split(':', 1)[0].strip(): line.split(':', 1)[1].strip()
                    for line in extracted_text.split('\n') if ':' in line
                }
                self.extracted_data['id_card'] = data_dict
                await self.view.send_extracted_text(update, "ID Card", extracted_text)
                await self.view.request_next_document(update, "Driver's License")
                return UPLOAD_LICENSE
            else:
                raise ValueError("Failed to extract data from ID Card")
        except asyncio.TimeoutError:
            logger.error("ID card processing timed out")
            await self.view.send_error_message(update, "Processing took too long. Please try again.")
            return UPLOAD_ID
        except Exception as e:
            logger.error(f"Error processing ID Card: {e}")
            await self.view.send_error_message(update, "Failed to process ID Card. Please try again.")
            return UPLOAD_ID
        finally:
            self.cleanup_temp_file(temp_path)

    async def handle_drivers_license(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        user = update.message.from_user
        photo = update.message.photo[-1]

        await self.view.send_processing_message(update, "Driver's License")
        temp_path = None

        try:
            saved_path, temp_path = await self.download_with_retry(photo, user.id, 'license')
            if not saved_path or not temp_path:
                raise ValueError("Failed to download image")

            is_valid, message = validate_image(saved_path)
            if not is_valid:
                logger.error(f"Image validation failed: {message}")
                await self.view.send_validation_error(update, message)
                return UPLOAD_LICENSE

            extracted_text = process_document(saved_path, 'license')
            data_dict = {
                line.split(':', 1)[0].strip(): line.split(':', 1)[1].strip()
                for line in extracted_text.split('\n') if ':' in line
            }
            self.extracted_data['license'] = data_dict
            await self.view.send_extracted_text(update, "Driver's License", extracted_text)
            await self.view.request_next_document(update, "Log Card")
            return UPLOAD_LOG
        except Exception as e:
            logger.error(f"Error processing Driver's License: {e}")
            await self.view.send_error_message(update, "Please try uploading the Driver's License photo again.")
            return UPLOAD_LICENSE
        finally:
            self.cleanup_temp_file(temp_path)

    async def handle_log_card(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        user = update.message.from_user
        photo = update.message.photo[-1]
        temp_path = None

        try:
            await self.view.send_processing_message(update, "Log Card")
            saved_path, temp_path = await self.download_with_retry(photo, user.id, 'log_card')
            
            if not saved_path or not temp_path:
                raise ValueError("Failed to download image")

            # Use saved_path (permanent path) for validation and processing
            is_valid, message = validate_image(saved_path)
            if not is_valid:
                logger.error(f"Image validation failed: {message}")
                await self.view.send_validation_error(update, message)
                return UPLOAD_LOG

            # Pass saved_path to process_document for processing
            extracted_text = process_document(saved_path, 'log_card')
            data_dict = {
                line.split(':', 1)[0].strip(): line.split(':', 1)[1].strip()
                for line in extracted_text.split('\n') if ':' in line
            }
            self.extracted_data['log_card'] = data_dict

            await self.view.send_extracted_text(update, "Log Card", extracted_text)
            return await self.handle_referrer_info(update, context)
        except Exception as e:
            logger.error(f"Error processing Log Card: {e}")
            await self.view.send_error_message(update, "Failed to process Log Card. Please try again.")
            return UPLOAD_LOG
        finally:
            self.cleanup_temp_file(temp_path)


    async def handle_referrer_info(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """
        Initiate referrer info collection by asking for Referrer's Name.
        """
        await self.view.send_message(update, "Enter Referrer's Name:")
        return STATE_REFERRER_NAME


    async def collect_referrer_name(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """
        Collect Referrer's Name and ask for Contact Number.
        """
        referrer_name = update.message.text.strip()
        if not referrer_name:
            await self.view.send_error_message(update, "Referrer's Name cannot be empty. Please try again.")
            return STATE_REFERRER_NAME

        # Save the collected name to context
        context.user_data['referrer_name'] = referrer_name
        await self.view.send_message(update, "Enter Contact Number:")
        return STATE_REFERRER_CONTACT


    async def collect_referrer_contact(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """
        Collect Contact Number and ask for Dealership.
        """
        contact_number = update.message.text.strip()
        if not contact_number:
            await self.view.send_error_message(update, "Contact Number cannot be empty. Please try again.")
            return STATE_REFERRER_CONTACT

        # Save the collected contact number to context
        context.user_data['contact_number'] = contact_number
        await self.view.send_message(update, "Enter Dealership:")
        return STATE_REFERRER_DEALERSHIP


    async def collect_referrer_dealership(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """
        Collect Dealership, validate inputs, and finalize the process.
        """
        dealership = update.message.text.strip()
        if not dealership:
            await self.view.send_error_message(update, "Dealership cannot be empty. Please try again.")
            return STATE_REFERRER_DEALERSHIP

        # Save the collected dealership to context
        context.user_data['dealership'] = dealership

        # Combine collected data into `self.extracted_data`
        self.extracted_data['referrer_info'] = {
            "Referrer's Name": context.user_data['referrer_name'],
            "Contact Number": context.user_data['contact_number'],
            "Dealership": context.user_data['dealership']
        }

        await self.view.send_message(update, "Referrer information collected successfully.")

        # Send data to Monday.com
        if not await self._send_to_monday():
            await self.view.send_data_save_error_message(update)
            return ConversationHandler.END

        await self.view.send_completion_message(update)
        return ConversationHandler.END


    async def _send_to_monday(self) -> bool:
        try:
            combined_data = {}
            combined_data.update(self.extracted_data.get('id_card', {}))
            combined_data.update(self.extracted_data.get('license', {}))
            combined_data.update(self.extracted_data.get('log_card', {}))
            combined_data.update(self.extracted_data.get('referrer_info', {}))
            return self.monday_service.create_policy_item(combined_data)
        except Exception as e:
            logger.error(f"Error sending data to Monday.com: {e}")
            return False

    async def prompt_user(self, update: Update, prompt_message: str) -> str:
        await self.view.send_message(update, prompt_message)
        message = await update.get_message()
        return message.text.strip()

    @staticmethod
    def cleanup_temp_file(temp_path: Optional[str]) -> None:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        await self.view.send_cancel_message(update)
        return ConversationHandler.END




