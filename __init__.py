"""
PlagiCheck IndoT5 Hybrid Paraphraser - Init File
Indonesian Text Paraphrasing using IndoT5 + Rule-based Hybrid Approach
"""

from .engines.indot5_hybrid_engine import IndoT5HybridParaphraser, IndoT5HybridResult
from .engines.quality_scorer import QualityScorer
from .utils.text_processor import TextProcessor
from .utils.validator import TextValidator

__version__ = "1.0.0"
__author__ = "devnolife"
__description__ = "Indonesian Text Paraphrasing using IndoT5 + Rule-based Hybrid Approach"

# Export main classes
__all__ = [
    'IndoT5HybridParaphraser',
    'IndoT5HybridResult', 
    'QualityScorer',
    'TextProcessor',
    'TextValidator'
]
