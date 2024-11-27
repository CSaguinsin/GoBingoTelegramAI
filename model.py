import pytesseract
from transformers import AutoProcessor
from PIL import Image, ImageEnhance
import torch
import cv2
import numpy as np
import logging
from transformers import AutoModelForVision2Seq
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Tesseract command path for macOS
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        gray = cv2.dilate(gray, kernel, iterations=1)
        pil_image = Image.fromarray(gray)
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(2)
        return pil_image
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        return None

def format_extracted_text(text):
    try:
        # Initialize default values
        formatted_data = {
            'name': '',
            'race': '',
            'dob': '',
            'sex': ''
        }
        
        # Split text into lines and clean them
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Extract information using pattern matching
        for i, line in enumerate(lines):
            # Name extraction
            if 'Name' in line and i + 1 < len(lines):
                name_parts = []
                next_line = lines[i + 1]
                if next_line and not any(keyword in next_line.upper() for keyword in ['RACE', 'DATE', 'SEX']):
                    name_parts.append(next_line.strip(' .'))
                    # Check for Chinese name in parentheses in the next line
                    if i + 2 < len(lines) and '(' in lines[i + 2] and ')' in lines[i + 2]:
                        chinese_name = lines[i + 2].strip()
                        name_parts.append(chinese_name)
                formatted_data['name'] = ' '.join(name_parts)
            
            # Race extraction
            elif 'Race' in line and i + 1 < len(lines):
                next_line = lines[i + 1]
                if next_line and not any(keyword in next_line.upper() for keyword in ['NAME', 'DATE', 'SEX']):
                    formatted_data['race'] = next_line.strip()
            
            # Date of birth extraction
            elif any(x in line for x in ['Date of birth', 'DOB']) and i + 1 < len(lines):
                next_line = lines[i + 1]
                # Clean up date format
                dob = next_line.replace('_', '').replace('LJ', '').replace('Mw', '').strip()
                if dob and len(dob) >= 8:  # Basic validation for date format
                    formatted_data['dob'] = dob
            
            # Sex extraction
            elif 'Sex' in line and i + 1 < len(lines):
                next_line = lines[i + 1].upper()
                if 'M' in next_line:
                    formatted_data['sex'] = 'M'
                elif 'F' in next_line:
                    formatted_data['sex'] = 'F'
        
        # Format the output string
        formatted_text = (
            f"Name: {formatted_data['name']}\n"
            f"Race: {formatted_data['race']}\n"
            f"Date of birth: {formatted_data['dob']}\n"
            f"Sex: {formatted_data['sex']}"
        )
        
        return formatted_text
    except Exception as e:
        logger.error(f"Error formatting text: {str(e)}")
        return text

def validate_image(image_path):
    try:
        with Image.open(image_path) as img:
            # Check if image is too small
            if img.size[0] < 100 or img.size[1] < 100:
                return False, "Image is too small"
            # Check if image is empty
            if os.path.getsize(image_path) < 1024:  # Less than 1KB
                return False, "Image file is too small"
            return True, "Image is valid"
    except Exception as e:
        return False, f"Invalid image: {str(e)}"

def extract_text_from_image(image_path):
    try:
        logger.info(f"Processing image: {image_path}")
        
        # Open and verify original image
        original_image = Image.open(image_path)
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')
        
        # Get OCR text
        try:
            processed_image = preprocess_image(image_path)
            if processed_image:
                logger.info("Using preprocessed image for OCR")
                ocr_text = pytesseract.image_to_string(processed_image)
            else:
                logger.info("Using original image for OCR")
                ocr_text = pytesseract.image_to_string(original_image)
            
            # Format the extracted text
            ocr_text = format_extracted_text(ocr_text)
            
        except Exception as e:
            logger.error(f"OCR Error: {str(e)}")
            ocr_text = "OCR processing failed"

        # Initialize VLM with error handling
        try:
            # Use SmolVLM with proper configuration
            processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
            model = AutoModelForVision2Seq.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
            
            device = torch.device("mps" if torch.backends.mps.is_available() else 
                                "cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            
            # Prepare a more detailed prompt for SmolVLM
            prompt = (
                "You are an ID card reader. Please extract and format the following information:\n"
                "1. Name (including Chinese name if present)\n"
                "2. Race\n"
                "3. Date of birth\n"
                "4. Sex\n"
                "Format the output exactly like this example:\n"
                "Name: [extracted name]\n"
                "Race: [extracted race]\n"
                "Date of birth: [extracted date]\n"
                "Sex: [extracted sex]"
            )
            
            # Process image and text with proper error checking
            inputs = processor(
                images=original_image,
                text=prompt,
                padding=True,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Generate with torch.no_grad() and optimized parameters for SmolVLM
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=256,
                    min_length=20,
                    num_beams=5,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True,
                    no_repeat_ngram_size=3,
                    length_penalty=1.0
                )
            
            # Add error checking for batch_decode
            decoded_outputs = processor.batch_decode(outputs, skip_special_tokens=True)
            
            if decoded_outputs and len(decoded_outputs[0].strip()) > 0:
                vlm_result = decoded_outputs[0]
                # If VLM result doesn't contain all required fields, combine with OCR
                if not all(field in vlm_result.lower() for field in ['name:', 'race:', 'date of birth:', 'sex:']):
                    logger.info("VLM output missing fields, combining with OCR results")
                    vlm_result = ocr_text
            else:
                logger.warning("VLM produced empty output, falling back to OCR")
                vlm_result = ocr_text
            
        except IndexError as e:
            logger.error(f"VLM Decoding Error: Output was empty - {str(e)}")
            vlm_result = ocr_text  # Fallback to OCR result
        except Exception as e:
            logger.error(f"VLM Error: {str(e)}")
            vlm_result = ocr_text  # Fallback to OCR result
        finally:
            # Cleanup
            if 'model' in locals():
                del model
            if 'processor' in locals():
                del processor
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return ocr_text, vlm_result
        
    except Exception as e:
        logger.error(f"General error: {str(e)}")
        return "Image processing failed", "Image processing failed"