from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
import logging

logger = logging.getLogger(__name__)

class ModelSingleton:
    _instance = None
    _initialized = False
    _model = None
    _processor = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelSingleton, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self._load_model()

    def _load_model(self):
        """Load the model once and cache it"""
        try:
            logger.info("Loading AI model...")
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self._device}")

            # Load processor and model only if not already loaded
            if self._processor is None:
                self._processor = AutoProcessor.from_pretrained(
                    "HuggingFaceTB/SmolVLM-Instruct",
                    trust_remote_code=True
                )
                logger.info("Processor loaded successfully")

            if self._model is None:
                self._model = AutoModelForVision2Seq.from_pretrained(
                    "HuggingFaceTB/SmolVLM-Instruct",
                    trust_remote_code=True
                )
                self._model.to(self._device)
                self._model.eval()  # Set to evaluation mode
                logger.info("Model loaded successfully")

        except Exception as e:
            self._initialized = False  # Reset initialization flag on failure
            logger.error(f"Failed to load model: {str(e)}")
            raise

    @property
    def model(self):
        return self._model

    @property
    def processor(self):
        return self._processor

    @property
    def device(self):
        return self._device

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelSingleton()
        return cls._instance

    def ensure_model_loaded(self):
        """Ensure model is loaded and optimize memory"""
        if self._model is None or self._processor is None:
            self._load_model()
        
        # Optimize memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Set model to evaluation mode
        if self._model is not None:
            self._model.eval()
  