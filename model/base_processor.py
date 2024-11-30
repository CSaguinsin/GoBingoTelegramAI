import pytesseract
from PIL import Image, ImageEnhance
import cv2
import torch
import logging
import os
from abc import ABC, abstractmethod
from transformers import AutoProcessor, AutoModelForVision2Seq
from typing import Tuple, Optional
from model.model_singleton import ModelSingleton

logger = logging.getLogger(__name__)

class BaseDocumentProcessor(ABC):
    def __init__(self):
        # Get singleton instance
        model_singleton = ModelSingleton.get_instance()
        
        # Get model, processor and device from singleton
        self.model = model_singleton.model
        self.processor = model_singleton.processor
        self.device = model_singleton.device
        
        # Set tesseract path
        pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
        
        # Should be set by child classes
        self.prompt = None

    def preprocess_image(self, image_path: str) -> Optional[Image.Image]:
        """Preprocess the image for better text extraction."""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
                
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to read image: {image_path}")
                
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Apply dilation
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
            gray = cv2.dilate(gray, kernel, iterations=1)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(gray)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(2)
            
            return pil_image
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            return None

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
            if hasattr(self, 'processor') and self.processor is not None:
                del self.processor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # MPS (Apple Silicon) cleanup
                torch.mps.empty_cache()
        except Exception as e:
            logger.error(f"Error in cleanup: {str(e)}")

    def verify_image(self, image_path: str) -> Optional[Image.Image]:
        """Verify and load image."""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
                
            original_image = Image.open(image_path)
            if original_image.mode != 'RGB':
                original_image = original_image.convert('RGB')
                
            return original_image
            
        except Exception as e:
            logger.error(f"Error verifying image: {str(e)}")
            return None

    @abstractmethod
    def process_image(self, image_path: str) -> Tuple[str, str]:
        """Process the image and extract text."""
        pass

    @abstractmethod
    def format_text(self, text: str) -> str:
        """Format the extracted text."""
        pass 