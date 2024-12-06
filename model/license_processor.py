from model.base_processor import BaseDocumentProcessor
import pytesseract
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class LicenseProcessor(BaseDocumentProcessor):
    def __init__(self):
        super().__init__()
        self.prompt = os.getenv('LICENSE_PROMPT')
        if not self.prompt:
            logger.error("LICENSE_PROMPT environment variable is required but not set")
            raise ValueError("LICENSE_PROMPT environment variable is required")


    def process_image(self, image_path):
        try:
            logger.info(f"Processing driver's license image: {image_path}")
            
            original_image = self.verify_image(image_path)
            if original_image is None:
                return "Image verification failed", "Image verification failed"
            
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
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=128,  # Reduced from 256
                        num_beams=2,         # Reduced from 3
                        temperature=0.3,
                        do_sample=True,
                        length_penalty=1.0,
                        repetition_penalty=1.2
                    )
                    
                    logger.info("Model inference completed, decoding output...")
                    generated_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
                    logger.info(f"Raw generated text: {generated_text}")
                    
                    formatted_text = self.format_text(generated_text)
                    logger.info(f"Formatted output: {formatted_text}")
                    
                    return formatted_text
                
            except Exception as e:
                logger.error(f"Generation error: {str(e)}")
                return "Text generation failed", "Text generation failed"
            finally:
                # Clean up CUDA memory
                self.cleanup()
                
        except Exception as e:
            logger.error(f"General error: {str(e)}")
            return "Image processing failed", "Image processing failed"

    def format_text(self, text: str) -> str:
        """Format the extracted text into a structured output."""
        try:
            # Initialize default values
            formatted = {
                "Name": "Not found",
                "License Number": "Not found",
                "Date of birth": "Not found",
                "Issue Date": "Not found"
            }
            
            # Extract information using simple pattern matching
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("Name:"):
                    formatted["Name"] = line.split(":", 1)[1].strip()
                elif line.startswith("License Number:"):
                    formatted["License Number"] = line.split(":", 1)[1].strip()
                elif line.startswith("Date of birth:"):
                    formatted["Date of birth"] = line.split(":", 1)[1].strip()
                elif line.startswith("Issue Date:"):
                    formatted["Issue Date"] = line.split(":", 1)[1].strip()
            
            # Format the output
            return "\n".join([f"{k}: {v}" for k, v in formatted.items()])
            
        except Exception as e:
            logger.error(f"Error formatting text: {str(e)}")
            return text