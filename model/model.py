import logging
from typing import Tuple
from .document_processor import DocumentProcessorFactory, DocumentProcessorError

logger = logging.getLogger(__name__)

def process_document(image_path: str, document_type: str = 'id_card') -> Tuple[str, str]:
    """
    Process a document image and extract text using OCR and VLM.
    
    Args:
        image_path (str): Path to the image file
        document_type (str): Type of document ('id_card', 'license', or 'log_card')
        
    Returns:
        Tuple[str, str]: A tuple containing (ocr_text, vlm_text)
        
    Raises:
        DocumentProcessorError: If document type is not supported or processing fails
        ValueError: If image_path is invalid
    """
    try:
        if not image_path:
            raise ValueError("Image path cannot be empty")
            
        processor = DocumentProcessorFactory.get_processor(document_type)
        return processor.process_image(image_path)
        
    except DocumentProcessorError as e:
        logger.error(f"Document processor error: {str(e)}")
        return f"Error: {str(e)}", f"Error: {str(e)}"
        
    except Exception as e:
        logger.error(f"Unexpected error processing document: {str(e)}")
        return "Document processing failed", "Document processing failed"