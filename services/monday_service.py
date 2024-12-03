import os
import requests
import logging
import json
from datetime import datetime
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

    def create_policy_item(self, data: dict) -> bool:
        try:
            logger.info("Preparing to create Monday.com item")
            board_id = str(self.board_id)
            
            logger.debug(f"Received data: {json.dumps(data, indent=2)}")
            
            # Format column values
            column_values = {}
            
            def format_text_value(value):
                if not value:
                    return {"text": ""}
                cleaned_value = str(value).strip()
                return {"text": cleaned_value}
            
            def format_date_value(value):
                formatted_date = self._format_date(value)
                return {"date": formatted_date} if formatted_date else {"date": ""}
            
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
                column_values[os.getenv('ROAD_TAX_EXPIRY_DATE', 'date57')] = format_text_value(data.get('Road Tax Expiry Date'))
            if data.get('PARF Eligibility Expiry Date'):
                column_values[os.getenv('PARF_ELIGIBILITY_EXPIRY_DATE', 'date44')] = format_date_value(data.get('PARF Eligibility Expiry Date'))
            if data.get('Inspection Due Date'):
                column_values[os.getenv('INSPECTION_DUE_DATE', 'date7')] = format_date_value(data.get('Inspection Due Date'))
            if data.get('Intended Transfer Date'):
                column_values[os.getenv('INTENDED_TRANSFER_DATE', 'date75')] = format_date_value(data.get('Intended Transfer Date'))
            
            # Add debug logging
            logger.debug(f"Raw column values: {json.dumps(column_values, indent=2)}")
            
            # Serialize column values to JSON string
            serialized_column_values = json.dumps(column_values)
            
            # GraphQL mutation
            mutation = """
            mutation ($boardId: ID!, $itemName: String!, $columnValues: JSON!) {
                create_item (
                    board_id: $boardId,
                    item_name: $itemName,
                    column_values: $columnValues
                ) {
                    id
                }
            }
            """
            
            variables = {
                "boardId": board_id,
                "itemName": f"{data.get('Name', '')} - {data.get('Vehicle No', 'New Policy')}".strip(),
                "columnValues": serialized_column_values
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
                "API-Version": "2024-01"
            }
            
            response = requests.post(
                self.api_url,
                json={"query": mutation, "variables": variables},
                headers=headers
            )
            
            response.raise_for_status()
            response_data = response.json()
            logger.debug(f"Monday.com API response: {json.dumps(response_data, indent=2)}")
            
            if "errors" in response_data:
                logger.error(f"Monday.com API error: {json.dumps(response_data['errors'], indent=2)}")
                return False
            
            if "data" in response_data and response_data["data"].get("create_item"):
                logger.info("Successfully created Monday.com item")
                return True
            
            logger.error(f"Unexpected response format from Monday.com: {json.dumps(response_data, indent=2)}")
            return False
            
        except Exception as e:
            logger.error(f"Error creating Monday.com item: {str(e)}")
            logger.exception(e)
            return False

    def _format_date(self, date_str: str) -> str:
        try:
            if not date_str or date_str == "0":
                return ""
            
            date_str = date_str.strip()
            
            date_formats = [
                "%d %b %Y",  # e.g., "27 Nov 2003"
                "%d/%m/%Y",  # e.g., "27/11/2003"
                "%Y-%m-%d",  # e.g., "2003-11-27"
            ]
            
            for date_format in date_formats:
                try:
                    parsed_date = datetime.strptime(date_str, date_format)
                    return parsed_date.strftime("%Y-%m-%d")  # Monday.com expects YYYY-MM-DD
                except ValueError:
                    continue
            
            logger.warning(f"Could not parse date: {date_str}")
            return ""
            
        except Exception as e:
            logger.error(f"Error formatting date: {str(e)}")
            return ""
