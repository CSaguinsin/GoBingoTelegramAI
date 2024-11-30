from model.base_processor import BaseDocumentProcessor
import pytesseract
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class LogCardProcessor(BaseDocumentProcessor):
    def __init__(self):
        super().__init__()
        self.prompt = os.getenv('LOG_CARD_PROMPT')
        if not self.prompt:
            logger.error("LOG_CARD_PROMPT environment variable is required but not set")
            raise ValueError("LOG_CARD_PROMPT environment variable is required")

    def process_image(self, image_path):
        try:
            logger.info(f"Processing log card image: {image_path}")
            
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
                    inputs = self.processor(
                        text=[self.prompt],
                        images=[original_image],
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
                
                # Initialize default values
                formatted = {
                    "Vehicle No": "Not found",
                    "Make/Model": "Not found",
                    "Chassis No": "Not found",
                    "Engine No": "Not found",
                    "Original Registration Date": "Not found",
                    "Engine Capacity": "Not found"
                }
                
                # Process each line
                for line in data_lines:
                    if line.startswith("Vehicle No:"):
                        formatted["Vehicle No"] = line.split(":", 1)[1].strip()
                    elif line.startswith("Make/Model:"):
                        formatted["Make/Model"] = line.split(":", 1)[1].strip()
                    elif line.startswith("Chassis No:"):
                        formatted["Chassis No"] = line.split(":", 1)[1].strip()
                    elif line.startswith("Engine No:"):
                        formatted["Engine No"] = line.split(":", 1)[1].strip()
                    elif line.startswith("Original Registration Date:"):
                        formatted["Original Registration Date"] = line.split(":", 1)[1].strip()
                    elif line.startswith("Engine Capacity:"):
                        formatted["Engine Capacity"] = line.split(":", 1)[1].strip()
                
                # Format the output
                return "\n".join([f"{k}: {v}" for k, v in formatted.items()])
                
            return "No data found"
                
        except Exception as e:
            logger.error(f"Error formatting text: {str(e)}")
            return text