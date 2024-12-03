from model.id_card_processor import IDCardProcessor
from model.license_processor import LicenseProcessor
from model.log_card_processor import LogCardProcessor
import logging

logger = logging.getLogger(__name__)

class DocumentProcessorFactory:
    @staticmethod
    def get_processor(document_type):
        if document_type == 'id_card':
            return IDCardProcessor()
        elif document_type == 'license':
            return LicenseProcessor()
        elif document_type == 'log_card':
            return LogCardProcessor()
        else:
            raise ValueError(f"Unsupported document type: {document_type}")

def process_document(image_path, document_type='id_card'):
    try:
        processor = DocumentProcessorFactory.get_processor(document_type)
        result = processor.process_image(image_path)
        
        # If result is a tuple, take the first element
        if isinstance(result, tuple):
            return result[0]
        return result
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return "Document processing failed" 