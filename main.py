import pytesseract
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image, ImageEnhance
import torch
import cv2
import numpy as np

# Set Tesseract command path for macOS
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

def preprocess_image(image_path):
    # Using existing preprocessing function
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    gray = cv2.dilate(gray, kernel, iterations=1)
    pil_image = Image.fromarray(gray)
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(2)
    return pil_image

def test_ocr(image_path):
    print(f"\nProcessing image: {image_path}")
    original_image = Image.open(image_path)
    processed_image = preprocess_image(image_path)
    
    # Basic OCR test with Tesseract
    print("\n=== Basic OCR Test ===")
    text = pytesseract.image_to_string(processed_image)
    print("\nOCR Result:")
    print("=" * 50)
    print(text)
    print("=" * 50)
    
    # SmolVLM test
    print("\n=== SmolVLM Test ===")
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
    model = AutoModelForVision2Seq.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
    
    # Move model to MPS device if available
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    # Process image with SmolVLM
    prompt = "Extract text."
    inputs = processor(images=original_image, text=prompt, return_tensors="pt")
    
    # Move inputs to device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    
    # Generate output
    outputs = model.generate(**inputs, max_length=512)
    result = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    
    print("\nSmolVLM Result:")
    print("=" * 50)
    print(result)
    print("=" * 50)

    return text, result

if __name__ == "__main__":
    # First verify the setup
    print("=== Verifying Setup ===")
    print("Tesseract Version:", pytesseract.get_tesseract_version())
    print("OpenCV Version:", cv2.__version__)
    
    # Use ID card image
    image_path = "/Users/carlsaginsin/Documents/Correct Set/IdentityCard_front.jpeg"
    print("\n=== Testing with ID Card image ===")
    ocr_text, vlm_result = test_ocr(image_path)