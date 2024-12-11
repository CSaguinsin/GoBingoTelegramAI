from model.base_processor import BaseDocumentProcessor
import logging
import os
logger = logging.getLogger(__name__)

class AgentProcessor(BaseDocumentProcessor):
    def __init__(self):
        super().__init__()
        self.prompt = os.getenv('AGENT_PROMPT')
        if not self.prompt:
            logger.error("AGENT_PROMPT environment variable is required but not set")
            raise ValueError("AGENT_PROMPT environment variable is required")

    def collect_referrer_info(self):
        """
        Simulate the process of collecting referrer information.
        This should be replaced with the actual interaction logic.
        """
        logger.info("Collecting referrer information.")
        referrer_name = input("Enter Referrer's Name: ")
        contact_number = input("Enter Contact Number: ")
        dealership = input("Enter Dealership: ")
        
        if not referrer_name or not contact_number or not dealership:
            logger.error("All fields are required.")
            raise ValueError("Referrer's Name, Contact Number, and Dealership are required.")
        
        logger.info("Referrer information collected successfully.")
        return {
            "Referrer's Name": referrer_name,
            "Contact Number": contact_number,
            "Dealership": dealership
        }
