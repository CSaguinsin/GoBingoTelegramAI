# GoBingo Telegram AI Document Processing Bot - Technical Documentation

## Table of Contents
1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Technical Components](#3-technical-components)
4. [Setup and Installation](#4-setup-and-installation)
5. [Configuration](#5-configuration)
6. [API Reference](#6-api-reference)
7. [Data Flow](#7-data-flow)
8. [Security Considerations](#8-security-considerations)
9. [Troubleshooting](#9-troubleshooting)
10. [Development Guidelines](#10-development-guidelines)
11. [Model Singleton Pattern Implementation](#11-model-singleton-pattern-implementation)

## 1. System Overview

### Purpose
GoBingo is an AI-powered Telegram bot designed to automate the processing and information extraction from identification documents. It supports multiple document types and uses advanced Vision Language Models (VLM) for accurate text extraction.

### Supported Document Types
- Identity Cards
- Driver's Licenses
- Vehicle Log Cards

### Key Features
- Real-time document processing
- Multi-stage validation
- Secure document handling
- Automated workflow management
- Error recovery and retry mechanisms

## 2. Architecture

### Design Pattern
The application follows the Model-View-Controller (MVC) pattern:
- **Model**: Document processing logic and AI models
- **View**: Telegram interface and user interactions
- **Controller**: Business logic and flow control

### Component Structure
```
├── bot.py                  # Application entry point
├── controller/             # Business logic layer
├── model/                  # Data and processing layer
├── view/                   # Presentation layer
```

## 3. Technical Components

### AI/ML Components
- **Model**: HuggingFaceTB/SmolVLM-Instruct
- **Framework**: PyTorch
- **Processing Pipeline**:
  1. Image Preprocessing & Validation
  2. Size Optimization
  3. VLM Analysis
  4. Text Extraction & Formatting
  5. Data Structuring
  6. Monday.com Integration

### Document Processors
Each document type has a dedicated processor inheriting from `BaseDocumentProcessor`:
```python
class BaseDocumentProcessor:
    - process_image()
    - format_text()
    - validate_document()
```

## 4. Setup and Installation

### System Requirements
- Python 3.8+
- 4GB RAM minimum
- 10GB storage space
- CUDA-compatible GPU (optional)

### Dependencies
```
python-telegram-bot==20.8
python-dotenv==1.0.0
torch
transformers
Pillow
opencv-python
pytesseract
```

### Installation Steps
1. **Environment Setup**
   ```bash
   git clone [repository-url]
   cd gobingo-telegram-bot
   python -m venv bingoenv
   source venv/bin/activate  # Unix/macOS
   pip install -r requirements.txt
   ```

2. **Model Installation**
   ```bash
   # The model will be automatically downloaded on first run
   # Default cache directory: ./model_cache/
   ```

## 5. Configuration

### Environment Variables
Create a `.env` file with:
```
TELEGRAM_BOT_API=your_bot_token
LICENSE_PROMPT=your_license_prompt
ID_CARD_PROMPT=your_id_card_prompt
LOG_CARD_PROMPT=your_log_card_prompt
```

### Directory Structure
```
├── image_documents/        # Permanent storage
│   ├── id_cards/
│   ├── licenses/
│   └── log_cards/
├── temp/                   # Temporary processing
└── model_cache/           # AI model cache
```

## 6. API Reference

### Controller Methods
```python
class TelegramController:
    - start()              # Initiates conversation
    - handle_id_card()     # Processes ID cards
    - handle_drivers_license()  # Processes licenses
    - handle_log_card()    # Processes log cards
    - cancel()             # Cancels current operation
```

### View Methods
```python
class TelegramView:
    - send_welcome_message()
    - send_processing_message()
    - send_error_message()
    - send_validation_error()
    - send_extracted_text()
    - request_next_document()
    - send_completion_message()
```

## 7. Data Flow

### Document Processing Pipeline
1. **Image Reception**
   - Telegram photo message received
   - Initial format validation

2. **Preprocessing**
   - Image size validation
   - Quality checks
   - Format standardization

3. **AI Processing**
   - OCR text extraction
   - VLM analysis
   - Data structuring

4. **Response Generation**
   - Text formatting
   - Error handling
   - User notification

## 8. Security Considerations

### Data Protection
- Temporary file cleanup
- Secure storage paths
- Environment variable protection

### Privacy Measures
- No permanent storage of sensitive data
- Automatic file deletion
- User data isolation

## 9. Troubleshooting

### Common Issues
1. **Image Processing Failures**
   - Check image quality
   - Verify file format
   - Ensure sufficient lighting

2. **Model Loading Issues**
   - Verify internet connection
   - Check disk space
   - Validate CUDA installation

3. **API Connection Errors**
   - Verify bot token
   - Check network connectivity
   - Validate API permissions

## 10. Development Guidelines

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Document all public methods

### Testing
```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/
```

## 11. Model Singleton Pattern Implementation

### Basic Structure
```python
class ModelSingleton:
    _instance = None
    _model = None
    _processor = None
    _device = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if ModelSingleton._instance is not None:
            raise Exception("This class is a singleton! Use get_instance() method.")
        else:
            ModelSingleton._instance = self
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._load_model()
```

### Model Loading Implementation
```python
    def _load_model(self):
        try:
            model_name = "HuggingFaceTB/SmolVLM-Instruct"
            cache_dir = "model_cache"
            
            # Ensure cache directory exists
            os.makedirs(cache_dir, exist_ok=True)
            
            # Load processor with caching
            self._processor = AutoProcessor.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            
            # Load model with appropriate settings
            self._model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).to(self._device)
            
            # Set model to evaluation mode
            self._model.eval()
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
```

### Usage Examples
```python
# Get model instance
model_instance = ModelSingleton.get_instance()

# Access model and processor
model = model_instance.model
processor = model_instance.processor
```

### Thread Safety Implementation
```python
from threading import Lock

class ModelSingleton:
    _instance = None
    _lock = Lock()
    
    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance
```

### Memory Management
```python
    def clear_cache(self):
        """
        Clears CUDA cache if using GPU.
        """
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self):
        """
        Cleanup when instance is deleted.
        """
        self.clear_cache()
```

## Contributing
1. Fork the repository
2. Create a feature branch
3. Follow coding standards
4. Write tests
5. Submit pull request

## License
[Your License Information Here]

## Support and Contact
- Technical Support: saguinsincarl8@gmail.com
- Bug Reports: GitHub Issues
- Feature Requests: GitHub Discussions
