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
            logger.info(f"Processing ID card image: {image_path}")
            
            original_image = self.verify_image(image_path)
            if original_image is None:
                return "Image verification failed", "Image verification failed"
            
            logger.info(f"Image opened successfully: {original_image.size}")
            
            try:
                self.processor, self.model = self.load_model()
                if self.processor is None or self.model is None:
                    return "Model loading failed", "Model loading failed"
                
                logger.info("Model loaded successfully")
                
                try:
                    # Prepare inputs for SmolVLM
                    inputs = self.processor(
                        text=[self.prompt],  # Wrap prompt in list
                        images=[original_image],  # Wrap image in list
                        return_tensors="pt",
                        padding=True
                    ).to(self.device)
                    
                    logger.info("Generating response...")
                    with torch.no_grad():
                        output_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=256,
                            num_beams=3,
                            temperature=0.3,
                            do_sample=False,
                            length_penalty=1.0,
                            repetition_penalty=1.2
                        )
                        
                        generated_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
                        logger.info(f"Raw generated text: {generated_text}")
                        
                        formatted_text = self.format_text(generated_text)
                        logger.info(f"Formatted output: {formatted_text}")
                        
                        return formatted_text, formatted_text
                    
                except Exception as e:
                    logger.error(f"Generation error: {str(e)}")
                    return "Text generation failed", "Text generation failed"
                    
            except Exception as e:
                logger.error(f"Model processing error: {str(e)}")
                return "Model processing failed", "Model processing failed"
                
            finally:
                self.cleanup()
                
        except Exception as e:
            logger.error(f"General error: {str(e)}")
            return "Image processing failed", "Image processing failed"

    def format_text(self, text: str) -> str:
        """Format the extracted text into a structured output."""
        try:
            # Split the text into lines and find where the actual data starts
            lines = text.split('\n')
            data_start = -1
            
            # Find where the actual data begins
            for i, line in enumerate(lines):
                if "Only output the extracted information" in line:
                    data_start = i + 2  # Skip the empty line after instructions
                    break
            
            if data_start >= 0 and data_start < len(lines):
                # Extract only the actual data lines, removing empty lines and cleaning up
                data_lines = [line.strip() for line in lines[data_start:] if line.strip()]
                
                # Join the cleaned data lines with newlines
                return '\n'.join(data_lines)
                
            return "No data found"
                
        except Exception as e:
            logger.error(f"Error formatting text: {str(e)}")
            return text