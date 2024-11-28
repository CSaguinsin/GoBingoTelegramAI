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
        self.prompt = (
            "Below is a driver's license image. <image>\n"
            "Extract and list the following information in exactly this format:\n"
            "Name: [Full name including Chinese name if present]\n"
            "License Number: [License number]\n"
            "Date of birth: [DOB in DD-MM-YYYY format]\n"
            "Issue Date: [Issue date in DD-MM-YYYY format]\n\n"
            "Only output the extracted information in the exact format above."
        )

    def process_image(self, image_path):
        try:
            logger.info(f"Processing driver's license image: {image_path}")
            
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
                        images=original_image,
                        text=self.prompt,
                        return_tensors="pt"
                    )
                    
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                    
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
                        
                        if output_ids is None or len(output_ids) == 0:
                            raise ValueError("Model generated empty output")
                        
                        generated_text = self.processor.batch_decode(output_ids, skip_special_tokens=True)
                        if not generated_text:
                            raise ValueError("Empty decoded outputs")
                            
                        vlm_result = generated_text[0]
                        logger.info(f"Raw VLM output: {vlm_result}")
                        
                        formatted_result = self.format_text(vlm_result)
                        logger.info(f"Formatted output: {formatted_result}")
                        
                        return formatted_result, formatted_result
                    
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