"""
Configuration for IndoT5 Hybrid Paraphraser
Contains settings for IndoT5 neural model and rule-based transformations
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

@dataclass
class IndoT5HybridConfig:
    """Configuration for IndoT5 Hybrid Paraphraser"""
    
    # IndoT5 Model settings
    model_name: str = "Wikidepia/IndoT5-base"
    use_gpu: bool = True
    max_length: int = 512
    num_beams: int = 4
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    
    # Quality thresholds
    min_quality_threshold: float = 60.0
    neural_confidence_threshold: float = 0.7
    semantic_similarity_threshold: float = 0.8
    
    # Synonym replacement settings
    synonym_replacement_rate: float = 0.3
    min_synonym_confidence: float = 0.6
    max_synonyms_per_word: int = 3
    preserve_entities: bool = True
    context_aware_synonyms: bool = True
    
    # Syntactic transformation settings
    max_transformations_per_sentence: int = 2
    transformation_types: List[str] = field(default_factory=lambda: [
        "active_to_passive",
        "clause_reordering",
        "conjunction_substitution", 
        "modifier_adjustment"
    ])
    
    # Quality scoring weights
    lexical_diversity_weight: float = 0.25
    semantic_preservation_weight: float = 0.35
    syntactic_complexity_weight: float = 0.20
    fluency_weight: float = 0.20
    
    # Processing settings
    max_processing_time: int = 30  # seconds
    max_paraphrase_attempts: int = 3
    enable_batch_processing: bool = True
    max_batch_size: int = 10
    
    # Data files
    synonym_file: str = "sinonim_extended.json"
    transformation_rules_file: str = "transformation_rules.json"
    stopwords_file: str = "stopwords_id.txt"
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "indot5_hybrid.log"

@dataclass
class SupportedModels:
    """Supported IndoT5 models configuration"""
    
    models: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "Wikidepia/IndoT5-base": {
            "size": "220M",
            "quality": "medium",
            "speed": "fast",
            "recommended": True
        },
        "Wikidepia/IndoT5-large": {
            "size": "770M", 
            "quality": "high",
            "speed": "slow",
            "recommended": False
        },
        "Wikidepia/IndoT5-small": {
            "size": "60M",
            "quality": "low",
            "speed": "very_fast", 
            "recommended": False
        }
    })

# Global configuration instances
config = IndoT5HybridConfig()
supported_models = SupportedModels()

class ConfigManager:
    """Manages IndoT5 Hybrid configuration"""
    
    def __init__(self, config_file: str = "indot5_hybrid_config.json"):
        self.config_file = BASE_DIR / config_file
        self.config = config
        self.supported_models = supported_models
    
    def load_config(self) -> bool:
        """Load configuration from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # Update config
                for key, value in config_data.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                
                return True
                
        except Exception as e:
            logging.warning(f"Failed to load config: {e}")
            return False
        
        return False
    
    def save_config(self) -> bool:
        """Save current configuration to file"""
        try:
            config_data = {
                key: value for key, value in self.config.__dict__.items() 
                if not key.startswith('_')
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to save config: {e}")
            return False
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        return self.supported_models.models.get(model_name, {})
    
    def get_recommended_model(self) -> str:
        """Get the recommended model"""
        for model_name, info in self.supported_models.models.items():
            if info.get("recommended", False):
                return model_name
        return "Wikidepia/IndoT5-base"

def setup_logging(log_level: str = None, log_file: str = None) -> None:
    """Setup logging configuration"""
    level = log_level or config.log_level
    file_path = log_file or (LOGS_DIR / config.log_file)
    
    # Create logs directory if it doesn't exist
    file_path.parent.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=config.log_format,
        handlers=[
            logging.FileHandler(file_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Set library log levels
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)

def get_data_path(filename: str) -> Path:
    """Get path to data file"""
    data_path = DATA_DIR / filename
    if data_path.exists():
        return data_path
    
    # Fallback to project root
    root_path = PROJECT_ROOT / filename
    if root_path.exists():
        return root_path
    
    return data_path

def validate_config() -> List[str]:
    """Validate configuration settings"""
    errors = []
    
    # Validate quality thresholds
    if not (0 <= config.min_quality_threshold <= 100):
        errors.append("min_quality_threshold must be between 0 and 100")
    
    if not (0 <= config.neural_confidence_threshold <= 1):
        errors.append("neural_confidence_threshold must be between 0 and 1")
    
    if not (0 <= config.semantic_similarity_threshold <= 1):
        errors.append("semantic_similarity_threshold must be between 0 and 1")
    
    # Validate synonym settings
    if not (0 <= config.synonym_replacement_rate <= 1):
        errors.append("synonym_replacement_rate must be between 0 and 1")
    
    if not (0 <= config.min_synonym_confidence <= 1):
        errors.append("min_synonym_confidence must be between 0 and 1")
    
    # Validate weights
    weight_sum = (config.lexical_diversity_weight + 
                  config.semantic_preservation_weight + 
                  config.syntactic_complexity_weight + 
                  config.fluency_weight)
    
    if abs(weight_sum - 1.0) > 0.01:
        errors.append("Quality scoring weights must sum to 1.0")
    
    # Validate model
    if config.model_name not in supported_models.models:
        errors.append(f"Unsupported model: {config.model_name}")
    
    # Validate data files
    required_files = [
        config.synonym_file,
        config.transformation_rules_file,
        config.stopwords_file
    ]
    
    for filename in required_files:
        if not get_data_path(filename).exists():
            errors.append(f"Required data file not found: {filename}")
    
    return errors

def create_default_config_file() -> bool:
    """Create default configuration file"""
    try:
        manager = ConfigManager()
        return manager.save_config()
    except Exception as e:
        logging.error(f"Failed to create default config: {e}")
        return False

# Initialize configuration
config_manager = ConfigManager()

# Try to load existing configuration
if not config_manager.load_config():
    # Create default configuration file
    create_default_config_file()

# Setup logging
setup_logging()

# Validate configuration
validation_errors = validate_config()
if validation_errors:
    logging.warning("Configuration validation errors:")
    for error in validation_errors:
        logging.warning(f"  - {error}")

# Export commonly used paths
SYNONYM_FILE_PATH = get_data_path(config.synonym_file)
TRANSFORMATION_RULES_PATH = get_data_path(config.transformation_rules_file)
STOPWORDS_FILE_PATH = get_data_path(config.stopwords_file)

__all__ = [
    'IndoT5HybridConfig',
    'SupportedModels',
    'ConfigManager',
    'config',
    'supported_models',
    'config_manager',
    'setup_logging',
    'get_data_path',
    'validate_config',
    'SYNONYM_FILE_PATH',
    'TRANSFORMATION_RULES_PATH',
    'STOPWORDS_FILE_PATH'
]
