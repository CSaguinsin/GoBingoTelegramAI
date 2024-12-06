from model.base_processor import BaseDocumentProcessor
import pytesseract
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class IDCardProcessor(BaseDocumentProcessor):
    def __init__(self):
        super().__init__()
        self.prompt = os.getenv('ID_CARD_PROMPT')
        if not self.prompt:
            logger.error("ID_CARD_PROMPT environment variable is required but not set")
            raise ValueError("ID_CARD_PROMPT environment variable is required")

    def process_image(self, image_path):
        try:
            logger.info(f"Processing image: {image_path}")
            
            original_image = self.verify_image(image_path)
            if original_image is None:
                return "Image verification failed"
            
            logger.info(f"Image opened successfully: {original_image.size}")
            
            try:
                # Optimize image size if too large
                max_size = 1024
                if original_image.size[0] > max_size or original_image.size[1] > max_size:
                    ratio = max_size / max(original_image.size)
                    new_size = tuple([int(dim * ratio) for dim in original_image.size])
                    original_image = original_image.resize(new_size, Image.Resampling.LANCZOS)
                    logger.info(f"Resized image to: {original_image.size}")

                logger.info("Preparing model inputs...")
                inputs = self.processor(
                    text=[self.prompt],
                    images=[original_image],
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                logger.info("Starting model inference...")
                with torch.no_grad():
                    # Set shorter max_new_tokens for faster processing
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=128,  # Reduced from 256
                        num_beams=2,         # Reduced from 3
                        temperature=0.3,
                        do_sample=True,
                        length_penalty=1.0,
                        repetition_penalty=1.2
                    )
                    
                    generated_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
                    logger.info(f"Raw generated text: {generated_text}")
                    
                    formatted_text = self.format_text(generated_text)
                    logger.info(f"Formatted output: {formatted_text}")
                    
                    return formatted_text
                
            except Exception as e:
                logger.error(f"Generation error: {str(e)}")
                return "Text generation failed"
            finally:
                # Clean up CUDA memory
                self.cleanup()
            
        except Exception as e:
            logger.error(f"General error: {str(e)}")
            return "Image processing failed"

    def format_text(self, text: str) -> str:
        try:
            # Initialize with required fields
            formatted_data = {
                "Name": "",
                "Race": "",
                "Date of birth": "",
                "Sex": "",
                "Country/Place of birth": "",
                "ID Number": ""
            }
            
            # Process lines
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if not line or "<image>" in line or "Extract only" in line:
                    continue
                    
                for field in formatted_data.keys():
                    if line.lower().startswith(field.lower() + ":"):
                        value = line.split(":", 1)[1].strip()
                        # Clean the value
                        value = value.replace('"', '')  # Remove quotes
                        value = value.replace('\n', ' ')  # Replace newlines
                        value = ' '.join(value.split())  # Normalize whitespace
                        formatted_data[field] = value
                        break
            
            # Format output
            output_lines = []
            for field, value in formatted_data.items():
                if value and value.lower() != "not found":  # Only include valid values
                    output_lines.append(f"{field}: {value}")
            
            return "\n".join(output_lines) if output_lines else "No data found"
            
        except Exception as e:
            logger.error(f"Error formatting text: {str(e)}")
            return "Error formatting text"