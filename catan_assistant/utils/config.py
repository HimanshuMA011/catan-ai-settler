"""
Configuration management for Catan Assistant.
"""

import os
import yaml
from typing import Dict, Any
from pathlib import Path

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from file or environment variables."""
    
    if config_path is None:
        config_path = os.getenv("CATAN_CONFIG_PATH", "config.yaml")
    
    # Default configuration
    default_config = {
        "llm": {
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.1,
            "max_tokens": 1000,
            "api_key": os.getenv("OPENAI_API_KEY")
        },
        "capture": {
            "default_region": {"x": 0, "y": 0, "width": 1920, "height": 1080},
            "capture_delay": 0.5
        },
        "extraction": {
            "ocr_engine": "tesseract",
            "vision_model": "yolo",
            "confidence_threshold": 0.7
        },
        "dspy": {
            "optimizer": "MIPROv2",
            "training_examples": 100,
            "validation_examples": 20
        },
        "memory": {
            "vector_db": "qdrant",
            "similarity_threshold": 0.75,
            "max_memories": 1000
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8000,
            "cors_origins": ["*"],
            "rate_limit": "100/minute"
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "catan_assistant.log"
        }
    }
    
    # Try to load from file
    if Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                default_config.update(file_config)
        except Exception as e:
            print(f"Warning: Could not load config file {config_path}: {e}")
    
    # Override with environment variables
    env_overrides = {
        "OPENAI_API_KEY": ("llm", "api_key"),
        "CATAN_LLM_MODEL": ("llm", "model_name"),
        "CATAN_API_PORT": ("api", "port"),
        "CATAN_LOG_LEVEL": ("logging", "level")
    }
    
    for env_var, (section, key) in env_overrides.items():
        if env_var in os.environ:
            if section not in default_config:
                default_config[section] = {}
            default_config[section][key] = os.environ[env_var]
    
    return default_config