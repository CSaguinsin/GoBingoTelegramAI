import os
import requests
import logging
import json
from datetime import datetime
import time
from typing import Dict

logger = logging.getLogger(__name__)

class MondayService:
    def __init__(self):
        self.api_token = os.getenv('MONDAY_API_TOKEN')
        self.api_url = os.getenv('MONDAY_API_URL', "https://api.monday.com/v2")
        self.board_id = os.getenv('POLICY_BOARD_ID')
        
        if not self.api_token or not self.board_id:
            logger.error("Missing required environment variables")
            raise ValueError("MONDAY_API_TOKEN and POLICY_BOARD_ID environment variables are required")

    def _validate_data(self, data: dict) -> bool:
        required_fields = ['Name']  # Add any other required fields
        
        for field in required_fields:
            if not data.get(field):
                logger.error(f"Missing required field: {field}")
                return False
                
        return True
    def create_policy_item(self, data: dict) -> bool:
        try:
            if not self._validate_data(data):
                return False
            
            max_retries = 3
            retry_delay = 1  # seconds
            
            for attempt in range(max_retries):
                try:
                    logger.info("Preparing to create Monday.com item")
                    board_id = str(self.board_id)
                    
                    logger.debug(f"Received data: {json.dumps(data, indent=2)}")
                    
                    # Validate required environment variables again
                    if not self.api_token or not self.board_id:
                        logger.error("Missing required API token or board ID")
                        return False

                    # Format column values
                    column_values = {}
                    
                    def format_text_value(value):
                        if not value:
                            return ""
                        return str(value).strip()
                    
                    def format_date_value(value):
                        formatted_date = self._format_date(value)
                        return formatted_date if formatted_date else ""
                    
                    # ID Card Data
                    if data.get('Name'):
                        column_values[os.getenv('FULL_NAME', 'text9')] = format_text_value(data.get('Name'))
                    if data.get('Date of birth'):
                        column_values[os.getenv('DATE_OF_BIRTH', 'text99')] = format_date_value(data.get('Date of birth'))
                    if data.get('Sex'):
                        column_values[os.getenv('SEX', 'text96')] = format_text_value(data.get('Sex'))
                    if data.get('Country/Place of birth'):
                        column_values[os.getenv('NATIONALITY', 'short_text')] = format_text_value(data.get('Country/Place of birth'))
                    if data.get('Race'):
                        column_values[os.getenv('RACE', 'text_17')] = format_text_value(data.get('Race'))
                    
                    # License Data
                    if data.get('License Number'):
                        column_values[os.getenv('LICENSE_NUMBER', 'text8')] = format_text_value(data.get('License Number'))
                    if data.get('Issue Date'):
                        column_values[os.getenv('ISSUE_DATE', 'date988')] = format_date_value(data.get('Issue Date'))
                    if data.get('Valid From'):
                        column_values[os.getenv('VALID_FROM', 'date4')] = format_date_value(data.get('Valid From'))
                    if data.get('Valid To'):
                        column_values[os.getenv('VALID_TO', 'date5')] = format_date_value(data.get('Valid To'))
                    if data.get('Classes'):
                        column_values[os.getenv('CLASSES', 'text_13')] = format_text_value(data.get('Classes'))
                    
                    # Vehicle Data
                    if data.get('Vehicle No'):
                        column_values[os.getenv('VEHICLE_NO', 'text_1195')] = format_text_value(data.get('Vehicle No'))
                    if data.get('Make/Model'):
                        make_model = data.get('Make/Model').split('/')
                        if len(make_model) > 0:
                            column_values[os.getenv('VEHICLE_MAKE', 'text2')] = format_text_value(make_model[0].strip())
                        if len(make_model) > 1:
                            column_values[os.getenv('VEHICLE_MODEL', 'text6')] = format_text_value(make_model[1].strip())
                    if data.get('Vehicle Type'):
                        column_values[os.getenv('VEHICLE_TYPE', 'text_1140')] = format_text_value(data.get('Vehicle Type'))
                    if data.get('Vehicle Attachment 1'):
                        column_values[os.getenv('VEHICLE_ATTACHMENT', 'text_18')] = format_text_value(data.get('Vehicle Attachment 1'))
                    if data.get('Vehicle Scheme'):
                        column_values[os.getenv('VEHICLE_SCHEME', 'text_157')] = format_text_value(data.get('Vehicle Scheme'))
                    if data.get('Chassis No'):
                        column_values[os.getenv('CHASSIS_NO', 'text775')] = format_text_value(data.get('Chassis No'))
                    if data.get('Propellant'):
                        column_values[os.getenv('PROPELLANT', 'text_153')] = format_text_value(data.get('Propellant'))
                    if data.get('Engine No'):
                        column_values[os.getenv('ENGINE_NUMBER', 'engine_number')] = format_text_value(data.get('Engine No'))
                    if data.get('Motor No'):
                        column_values[os.getenv('MOTOR_NO', 'text_155')] = format_text_value(data.get('Motor No'))
                    if data.get('Engine Capacity'):
                        column_values[os.getenv('ENGINE_CAPACITY', 'text_12')] = format_text_value(data.get('Engine Capacity'))
                    if data.get('Power Rating'):
                        column_values[os.getenv('POWER_RATING', 'text_156')] = format_text_value(data.get('Power Rating'))
                    if data.get('Maximum Power Output'):
                        column_values[os.getenv('MAXIMUM_POWER_OUTPUT', 'text_10')] = format_text_value(data.get('Maximum Power Output'))
                    if data.get('Maximum Laden Weight'):
                        column_values[os.getenv('MAXIMUM_LADEN_WEIGHT', 'text_15')] = format_text_value(data.get('Maximum Laden Weight'))
                    if data.get('Unladen Weight'):
                        column_values[os.getenv('UNLADEN_WEIGHT', 'text_14')] = format_text_value(data.get('Unladen Weight'))
                    if data.get('Year Of Manufacture'):
                        column_values[os.getenv('YEAR_OF_MANUFACTURE', 'text_11')] = format_text_value(data.get('Year Of Manufacture'))
                    if data.get('COE Category'):
                        column_values[os.getenv('COE_CATEGORY', 'text_171')] = format_text_value(data.get('COE Category'))
                    if data.get('PQP Paid'):
                        column_values[os.getenv('PQP_PAID', 'text_114')] = format_text_value(data.get('PQP Paid'))

                    # Date fields
                    if data.get('Original Registration Date'):
                        column_values[os.getenv('ORIGINAL_REGISTRATION_DATE', 'date8')] = format_date_value(data.get('Original Registration Date'))
                    if data.get('COE Expiry Date'):
                        column_values[os.getenv('COE_EXPIRY_DATE', 'date1')] = format_date_value(data.get('COE Expiry Date'))
                    if data.get('Road Tax Expiry Date'):
                        column_values[os.getenv('ROAD_TAX_EXPIRY_DATE', 'date57')] = format_date_value(data.get('Road Tax Expiry Date'))
                    if data.get('PARF Eligibility Expiry Date'):
                        column_values[os.getenv('PARF_ELIGIBILITY_EXPIRY_DATE', 'date44')] = format_date_value(data.get('PARF Eligibility Expiry Date'))
                    if data.get('Inspection Due Date'):
                        column_values[os.getenv('INSPECTION_DUE_DATE', 'date7')] = format_date_value(data.get('Inspection Due Date'))
                    if data.get('Intended Transfer Date'):
                        column_values[os.getenv('INTENDED_TRANSFER_DATE', 'date75')] = format_date_value(data.get('Intended Transfer Date'))

                    # Add Referrer Information to column values
                    if data.get("Referrer's Name"):
                        column_values[os.getenv('REFERRER_NAME', 'text23')] = format_text_value(data.get("Referrer's Name"))
                    if data.get("Contact Number"):
                        column_values[os.getenv('CONTACT_NUMBER', 'phone0')] = format_text_value(data.get("Contact Number"))
                    if data.get("Dealership"):
                        column_values[os.getenv('DEALERSHIP', 'text3')] = format_text_value(data.get("Dealership"))

                    # Convert the column values to Monday.com's expected format
                    formatted_values = {}
                    for key, value in column_values.items():
                        if isinstance(value, str):  # Handle string values
                            if key.startswith('date'):
                                formatted_values[key] = {"date": value}
                            else:
                                formatted_values[key] = {"text": value}
                        elif isinstance(value, dict):  # Handle already formatted values
                            formatted_values[key] = value
                        else:  # Handle any other type
                            formatted_values[key] = {"text": str(value)}

                    # Validate formatted values before sending
                    if not formatted_values:
                        logger.error("No valid column values to send")
                        return False

                    # Before sending the request
                    logger.debug("Formatted column values:")
                    for key, value in formatted_values.items():
                        logger.debug(f"{key}: {value}")

                    # GraphQL mutation with better formatting
                    mutation = """
                    mutation createItem ($boardId: ID!, $itemName: String!, $columnValues: JSON!) {
                        create_item (
                            board_id: $boardId,
                            item_name: $itemName,
                            column_values: $columnValues
                        ) {
                            id
                        }
                    }
                    """

                    # Ensure item name is not empty
                    item_name = f"{data.get('Name', '')} - {data.get('Vehicle No', 'New Policy')}".strip()
                    if not item_name:
                        item_name = "New Policy"

                    # Convert column_values to JSON string as required by Monday.com API
                    formatted_values = json.dumps(column_values)

                    variables = {
                        "boardId": board_id,
                        "itemName": item_name,
                        "columnValues": formatted_values
                    }

                    headers = {
                        "Authorization": f"Bearer {self.api_token}",
                        "Content-Type": "application/json",
                        "API-Version": "2024-01",
                        "Accept": "application/json"
                    }

                    # Add request timeout and better error handling
                    try:
                        response = requests.post(
                            self.api_url,
                            json={"query": mutation, "variables": variables},
                            headers=headers,
                            timeout=30
                        )
                        
                        if response.status_code == 429:  # Rate limit
                            wait_time = int(response.headers.get('Retry-After', retry_delay * (2 ** attempt)))
                            logger.warning(f"Rate limited. Waiting {wait_time} seconds...")
                            time.sleep(wait_time)
                            continue
                        
                        # Log the complete request details for debugging
                        logger.debug("Request details:")
                        logger.debug(f"URL: {self.api_url}")
                        logger.debug(f"Headers: {self._safe_json_dumps(headers)}")
                        logger.debug(f"Payload: {self._safe_json_dumps({'query': mutation, 'variables': variables})}")
                        
                        # Handle different response status codes
                        if response.status_code == 401:
                            logger.error("Authentication failed. Check your API token.")
                            return False
                        elif response.status_code == 400:
                            logger.error(f"Bad request: {response.text}")
                            return False
                        elif response.status_code != 200:
                            logger.error(f"Unexpected status code {response.status_code}: {response.text}")
                            return False

                        response_data = response.json()
                        
                        # Enhanced error checking
                        if "errors" in response_data:
                            error_messages = [error.get('message', 'Unknown error') 
                                            for error in response_data.get('errors', [])]
                            logger.error(f"Monday.com API errors: {', '.join(error_messages)}")
                            return False
                        
                        if "data" in response_data and response_data["data"].get("create_item"):
                            logger.info("Successfully created Monday.com item")
                            return True
                        
                        logger.error(f"Unexpected response format: {json.dumps(response_data, indent=2)}")
                        return False

                    except requests.exceptions.RequestException as e:
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (2 ** attempt)
                            logger.warning(f"Request failed, retrying in {wait_time} seconds... ({str(e)})")
                            time.sleep(wait_time)
                            continue
                        logger.error(f"Request error after {max_retries} attempts: {str(e)}")
                        return False
                    
                except Exception as e:
                    logger.error(f"Error in attempt {attempt + 1}: {str(e)}")
                    if attempt < max_retries - 1:
                        continue
                    raise
        except Exception as e:
            logger.error(f"Error creating Monday.com item: {str(e)}")
            logger.exception(e)
            return False

    def _format_date(self, date_str: str) -> str:
        try:
            if not date_str or date_str == "0" or date_str.lower() in ["not found", "-"]:
                return ""
            
            date_str = date_str.strip()
            
            # Try parsing with dateutil first
            try:
                from dateutil import parser
                parsed_date = parser.parse(date_str)
                return parsed_date.strftime("%Y-%m-%d")
            except:
                # Additional date formats
                date_formats = [
                    "%d-%m-%Y",      # e.g., "22-06-1971"
                    "%d %b %Y",      # e.g., "22 Jun 1971"
                    "%d-%b-%Y",      # e.g., "22-Jun-1971"
                    "%d/%m/%Y",      # e.g., "22/06/1971"
                    "%Y-%m-%d",      # e.g., "1971-06-22"
                    "%d %B %Y",      # e.g., "22 June 1971"
                    "%B %d, %Y",     # e.g., "June 22, 1971"
                ]
                
                for date_format in date_formats:
                    try:
                        parsed_date = datetime.strptime(date_str, date_format)
                        return parsed_date.strftime("%Y-%m-%d")
                    except ValueError:
                        continue
            
            logger.warning(f"Could not parse date: {date_str}")
            return ""
            
        except Exception as e:
            logger.error(f"Error formatting date: {str(e)}")
            return ""

    def _safe_json_dumps(self, obj):
        try:
            return json.dumps(obj)
        except Exception as e:
            logger.error(f"Error encoding JSON: {str(e)}")
            return json.dumps({})