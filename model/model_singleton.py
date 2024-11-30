from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
import os
import logging

logger = logging.getLogger(__name__)

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
            raise Exception("This class is a singleton!")
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._load_model()

    def _load_model(self):
        try:
            model_name = "HuggingFaceTB/SmolVLM-Instruct"
            
            logger.info("Loading model and processor...")
            self._processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            self._model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).to(self._device)
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
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