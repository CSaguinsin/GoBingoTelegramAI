from telegram import Update, File
from telegram.ext import ContextTypes, ConversationHandler
from model import process_document, validate_image
from view.view import TelegramView
from telegram.error import TimedOut, NetworkError
import os
import time
import logging
import asyncio
from typing import Tuple, Optional
from datetime import datetime
import shutil
from services.monday_service import MondayService

logger = logging.getLogger(__name__)

# Define conversation states
UPLOAD_ID, UPLOAD_LICENSE, UPLOAD_LOG = range(3)
MAX_RETRIES = 3
RETRY_DELAY = 2
class TelegramController:
    def __init__(self) -> None:
        self.view = TelegramView()
        self.monday_service = MondayService()
        self.extracted_data = {}
    async def download_with_retry(self, photo, user_id: int, doc_type: str, max_retries: int = 3) -> Tuple[str, str]:
        """Download photo with retry mechanism and save to both temp and permanent locations."""
        for attempt in range(max_retries):
            try:
                # Create directories if they don't exist
                temp_dir = os.path.join('temp_documents')
                perm_dir = os.path.join('image_documents', doc_type)
                os.makedirs(temp_dir, exist_ok=True)
                os.makedirs(perm_dir, exist_ok=True)

                # Generate unique filenames
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{user_id}_{timestamp}.jpg"
                temp_path = os.path.join(temp_dir, filename)
                saved_path = os.path.join(perm_dir, filename)

                # Download file
                file = await photo.get_file()
                await file.download_to_drive(temp_path)

                # Copy to permanent storage
                shutil.copy2(temp_path, saved_path)

                logger.info(f"Successfully downloaded and saved image. Temp: {temp_path}, Saved: {saved_path}")
                return saved_path, temp_path

            except Exception as e:
                logger.error(f"Download attempt {attempt + 1} failed: {str(e)}")
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
            
            try:
                # Process document with longer timeout
                extracted_text = await asyncio.wait_for(
                    asyncio.to_thread(process_document, saved_path, 'id_card'),
                    timeout=300.0  # 5 minutes timeout
                )
                
                if extracted_text and "No data found" not in extracted_text:
                    # Parse the extracted text into a dictionary
                    data_dict = {}
                    for line in extracted_text.split('\n'):
                        if ':' in line:
                            key, value = line.split(':', 1)
                            data_dict[key.strip()] = value.strip()
                    self.extracted_data['id_card'] = data_dict
                    await self.view.send_extracted_text(update, "ID Card", extracted_text)
                    await self.view.request_next_document(update, "Driver's License")
                    
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    return UPLOAD_LICENSE
                else:
                    raise ValueError("Failed to extract data from ID Card")
                    
            except asyncio.TimeoutError:
                logger.error("ID card processing timed out")
                await self.view.send_error_message(update, "Processing took too long. Please try again.")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return UPLOAD_ID
                
        except Exception as e:
            logger.error(f"Error processing ID Card: {str(e)}")
            await self.view.send_error_message(update, "Failed to process ID Card. Please try again.")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return UPLOAD_ID

    async def handle_drivers_license(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        user = update.message.from_user
        photo = update.message.photo[-1]
        
        await self.view.send_processing_message(update, "Driver's License")
        
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
            # Parse the extracted text into a dictionary
            data_dict = {}
            for line in extracted_text.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    data_dict[key.strip()] = value.strip()
            self.extracted_data['license'] = data_dict
            await self.view.send_extracted_text(update, "Driver's License", extracted_text)
            await self.view.request_next_document(update, "Log Card")
            
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            return UPLOAD_LOG
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            await self.view.send_error_message(update, "Please try uploading the Driver's License photo again.")
            return UPLOAD_LICENSE

    async def handle_log_card(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        user = update.message.from_user
        photo = update.message.photo[-1]
        temp_path = None
        
        try:
            # Wrap the processing message in try-except
            try:
                await self.view.send_processing_message(update, "Log Card")
            except TimedOut:
                logger.warning("Timeout sending processing message, continuing anyway")
            
            saved_path, temp_path = await self.download_with_retry(photo, user.id, 'log_card')
            if not saved_path or not temp_path:
                raise ValueError("Failed to download image")
            
            is_valid, message = validate_image(saved_path)
            if not is_valid:
                logger.error(f"Image validation failed: {message}")
                await self.view.send_validation_error(update, message)
                return UPLOAD_LOG
            
            extracted_text = process_document(saved_path, 'log_card')
            # Parse the extracted text into a dictionary
            data_dict = {}
            for line in extracted_text.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    data_dict[key.strip()] = value.strip()
            self.extracted_data['log_card'] = data_dict
            
            # Send data to Monday.com
            if await self._send_to_monday():
                await self.view.send_data_saved_message(update)
            else:
                await self.view.send_data_save_error_message(update)
            
            await self.view.send_extracted_text(update, "Log Card", extracted_text)
            await self.view.send_completion_message(update)
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            self.extracted_data = {}
                
            return ConversationHandler.END
            
        except TimedOut:
            logger.error("Connection timed out")
            await asyncio.sleep(2)  # Wait before retrying
            await self.view.send_error_message(
                update, 
                "Connection timed out. Please try again."
            )
            return UPLOAD_LOG
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            await self.view.send_error_message(
                update, 
                "Please try uploading the Log Card photo again."
            )
            return UPLOAD_LOG
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        await self.view.send_cancel_message(update)
        return ConversationHandler.END



    async def _send_to_monday(self) -> bool:
        """Combine all data and send to Monday.com."""
        try:
            # Log the data being combined for debugging
            logger.debug("ID Card data: %s", self.extracted_data.get('id_card', {}))
            logger.debug("License data: %s", self.extracted_data.get('license', {}))
            logger.debug("Log Card data: %s", self.extracted_data.get('log_card', {}))
            
            combined_data = {}
            # Update in specific order to ensure correct data precedence
            combined_data.update(self.extracted_data.get('id_card', {}))
            combined_data.update(self.extracted_data.get('license', {}))
            combined_data.update(self.extracted_data.get('log_card', {}))
            
            logger.debug("Combined data being sent to Monday.com: %s", combined_data)
            return self.monday_service.create_policy_item(combined_data)
            
        except Exception as e:
            logger.error(f"Error sending data to Monday.com: {str(e)}")
            return False

