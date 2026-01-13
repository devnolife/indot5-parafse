"""
IndoT5 Hybrid Paraphraser Engine
Combines IndoT5 neural processing with rule-based transformations
Following hybrid approach: Neural generation -> Rule-based enhancement
"""

import json
import os
import re
import random
import logging
import time
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IndoT5HybridResult:
    """Result container for IndoT5 hybrid paraphrasing"""
    original_text: str
    paraphrased_text: str
    method_used: str
    transformations_applied: List[str]
    quality_score: float
    confidence_score: float
    neural_confidence: float
    semantic_similarity: float
    lexical_diversity: float
    syntactic_complexity: float
    fluency_score: float
    processing_time: float
    word_changes: int
    syntax_changes: int
    success: bool = True
    error_message: Optional[str] = None
    alternatives: List[str] = field(default_factory=list)

class IndoT5HybridParaphraser:
    """
    IndoT5 Hybrid Paraphraser Engine
    
    Combines IndoT5 neural processing with rule-based transformations:
    1. IndoT5 neural generation for initial paraphrase
    2. Rule-based enhancement (synonym substitution, syntactic transformation)
    3. Quality assessment and validation
    """
    
    def __init__(self, 
                 model_name: str = "Wikidepia/IndoT5-base",
                 use_gpu: bool = True,
                 synonym_rate: float = 0.3,
                 min_confidence: float = 0.7,
                 quality_threshold: float = 75.0,
                 max_transformations: int = 3,
                 enable_caching: bool = True):
        """
        Initialize IndoT5 Hybrid Paraphraser
        
        Args:
            model_name: IndoT5 model name
            use_gpu: Whether to use GPU acceleration
            synonym_rate: Synonym replacement rate (0.0-1.0)
            min_confidence: Minimum neural confidence threshold
            quality_threshold: Minimum quality score threshold
            max_transformations: Maximum rule-based transformations
            enable_caching: Enable model and result caching
        """
        self.model_name = model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.synonym_rate = synonym_rate
        self.min_confidence = min_confidence
        self.quality_threshold = quality_threshold
        self.max_transformations = max_transformations
        self.enable_caching = enable_caching
        
        # Initialize device
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        
        # Initialize models
        self._init_models()
        
        # Load data
        self._load_data()
        
        # Initialize caches
        if self.enable_caching:
            self._result_cache = {}
            self._synonym_cache = {}
        
        logger.info(f"✅ IndoT5 Hybrid Paraphraser initialized")
        logger.info(f"   Model: {self.model_name}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   GPU: {self.use_gpu}")
    
    def _init_models(self):
        """Initialize IndoT5 and semantic similarity models"""
        try:
            # Load IndoT5 model
            logger.info(f"🔄 Loading IndoT5 model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            if self.use_gpu:
                self.model = self.model.to(self.device)
            
            # Load semantic similarity model
            logger.info("🔄 Loading semantic similarity model")
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            logger.info("✅ Models loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            raise
    
    def _load_data(self):
        """Load synonym database and transformation rules"""
        try:
            # Load synonym database
            current_dir = os.path.dirname(os.path.abspath(__file__))
            synonym_path = os.path.join(current_dir, "..", "data", "sinonim_extended.json")
            
            if os.path.exists(synonym_path):
                with open(synonym_path, 'r', encoding='utf-8') as f:
                    self.synonym_data = json.load(f)
                logger.info(f"✅ Loaded {len(self.synonym_data)} synonyms")
            else:
                logger.warning("⚠️  Synonym file not found, using empty dictionary")
                self.synonym_data = {}
            
            # Load transformation rules
            rules_path = os.path.join(current_dir, "..", "data", "transformation_rules.json")
            
            if os.path.exists(rules_path):
                with open(rules_path, 'r', encoding='utf-8') as f:
                    self.transformation_rules = json.load(f)
                logger.info("✅ Loaded transformation rules")
            else:
                logger.warning("⚠️  Transformation rules not found, using defaults")
                self.transformation_rules = self._get_default_rules()
            
            # Load stopwords
            stopwords_path = os.path.join(current_dir, "..", "data", "stopwords_id.txt")
            
            if os.path.exists(stopwords_path):
                with open(stopwords_path, 'r', encoding='utf-8') as f:
                    self.stop_words = set(line.strip() for line in f.readlines())
                logger.info(f"✅ Loaded {len(self.stop_words)} stopwords")
            else:
                logger.warning("⚠️  Stopwords file not found, using defaults")
                self.stop_words = self._get_default_stopwords()
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            # Use default data
            self.synonym_data = {}
            self.transformation_rules = self._get_default_rules()
            self.stop_words = self._get_default_stopwords()
    
    def _get_default_rules(self) -> Dict[str, List[Dict[str, str]]]:
        """Get default transformation rules"""
        return {
            "active_to_passive": [
                {"pattern": r"(\w+)\s+(me\w+)\s+(\w+)", "replacement": r"\3 di\2 oleh \1"},
                {"pattern": r"(\w+)\s+(akan|telah)\s+(\w+)\s+(\w+)", "replacement": r"\4 \2 di\3 oleh \1"}
            ],
            "conjunction_substitution": [
                {"pattern": r"\bdan\b", "replacement": "serta"},
                {"pattern": r"\btetapi\b", "replacement": "namun"},
                {"pattern": r"\bkarena\b", "replacement": "sebab"},
                {"pattern": r"\bjika\b", "replacement": "apabila"}
            ],
            "modifier_adjustment": [
                {"pattern": r"\bsangat\s+(\w+)", "replacement": r"amat \1"},
                {"pattern": r"\bcukup\s+(\w+)", "replacement": r"agak \1"}
            ]
        }
    
    def _get_default_stopwords(self) -> set:
        """Get default Indonesian stopwords"""
        return {
            'dan', 'atau', 'yang', 'adalah', 'dengan', 'untuk', 'dalam', 'dari', 
            'pada', 'ke', 'di', 'ini', 'itu', 'akan', 'dapat', 'juga', 'tidak',
            'ada', 'sudah', 'telah', 'harus', 'bisa', 'lebih', 'sangat', 'saya',
            'kami', 'kita', 'mereka', 'dia', 'ia', 'anda', 'kamu', 'maka',
            'oleh', 'bila', 'jika', 'ketika', 'saat', 'waktu', 'dimana', 'bagaimana',
            'mengapa', 'kenapa', 'siapa', 'apa', 'kapan', 'namun', 'tetapi',
            'karena', 'sebab', 'sehingga', 'meski', 'walaupun', 'meskipun'
        }
    
    def _neural_paraphrase(self, text: str, num_beams: int = 4, temperature: float = 0.8) -> Tuple[str, float]:
        """
        Generate paraphrase using IndoT5 neural model
        
        Args:
            text: Input text
            num_beams: Number of beams for beam search
            temperature: Temperature for generation
            
        Returns:
            Tuple of (paraphrased_text, confidence_score)
        """
        try:
            # Prepare input
            input_text = f"paraphrase: {text}"
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            if self.use_gpu:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=num_beams,
                    temperature=temperature,
                    do_sample=True,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode
            paraphrased = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Calculate confidence (simplified)
            confidence = min(1.0, len(paraphrased.split()) / len(text.split()) * 0.8 + 0.2)
            
            return paraphrased, confidence
            
        except Exception as e:
            logger.error(f"❌ Neural paraphrase failed: {e}")
            return text, 0.0
    
    def _apply_synonym_substitution(self, text: str, rate: float = None) -> Tuple[str, List[str], int]:
        """
        Apply synonym substitution to text
        
        Args:
            text: Input text
            rate: Synonym replacement rate
            
        Returns:
            Tuple of (modified_text, transformations, changes_count)
        """
        if rate is None:
            rate = self.synonym_rate
        
        words = text.split()
        result = []
        transformations = []
        changes_count = 0
        
        for word in words:
            clean_word = word.lower().strip('.,!?;:"')
            
            # Skip stopwords and short words
            if clean_word in self.stop_words or len(clean_word) <= 2:
                result.append(word)
                continue
            
            # Check if word has synonyms
            if clean_word in self.synonym_data and random.random() < rate:
                # Handle both list format and dict format
                syn_data = self.synonym_data[clean_word]
                if isinstance(syn_data, list):
                    synonyms = syn_data
                elif isinstance(syn_data, dict):
                    synonyms = syn_data.get('sinonim', [])
                else:
                    synonyms = []
                
                if synonyms:
                    # Choose random synonym
                    chosen_synonym = random.choice(synonyms)
                    
                    # Preserve original word format
                    if word.isupper():
                        formatted_synonym = chosen_synonym.upper()
                    elif word.istitle():
                        formatted_synonym = chosen_synonym.title()
                    else:
                        formatted_synonym = chosen_synonym
                    
                    # Preserve punctuation
                    if word[-1] in '.,!?;:"':
                        formatted_synonym += word[-1]
                    
                    result.append(formatted_synonym)
                    transformations.append(f"synonym: {word} -> {formatted_synonym}")
                    changes_count += 1
                else:
                    result.append(word)
            else:
                result.append(word)
        
        return ' '.join(result), transformations, changes_count
    
    def _apply_syntactic_transformation(self, text: str, max_transforms: int = None) -> Tuple[str, List[str], int]:
        """
        Apply syntactic transformations to text
        
        Args:
            text: Input text
            max_transforms: Maximum number of transformations
            
        Returns:
            Tuple of (modified_text, transformations, changes_count)
        """
        if max_transforms is None:
            max_transforms = self.max_transformations
        
        result = text
        transformations = []
        changes_count = 0
        
        # Available transformation types
        transform_types = [
            "active_to_passive",
            "conjunction_substitution", 
            "modifier_adjustment"
        ]
        
        # Apply random transformations
        num_transforms = min(max_transforms, len(transform_types))
        selected_transforms = random.sample(transform_types, num_transforms)
        
        for transform_type in selected_transforms:
            if transform_type in self.transformation_rules:
                rules_data = self.transformation_rules[transform_type]
                
                # Handle list format (active_to_passive, modifier_adjustment)
                if isinstance(rules_data, list):
                    for rule in rules_data:
                        pattern = rule.get("pattern", "")
                        replacement = rule.get("replacement", "")
                        
                        if pattern and re.search(pattern, result, re.IGNORECASE):
                            new_result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
                            
                            if new_result != result:
                                transformations.append(f"syntactic: {transform_type}")
                                changes_count += 1
                                result = new_result
                                break  # Only apply one rule per type
                
                # Handle dict format (conjunction_substitution)
                elif isinstance(rules_data, dict):
                    for word, word_data in rules_data.items():
                        alternatives = word_data.get("alternatives", []) if isinstance(word_data, dict) else []
                        if alternatives and re.search(rf'\b{word}\b', result, re.IGNORECASE):
                            chosen = random.choice(alternatives)
                            new_result = re.sub(rf'\b{word}\b', chosen, result, count=1, flags=re.IGNORECASE)
                            
                            if new_result != result:
                                transformations.append(f"syntactic: {transform_type} ({word} -> {chosen})")
                                changes_count += 1
                                result = new_result
                                break
        
        return result, transformations, changes_count
    
    def _calculate_quality_metrics(self, original: str, paraphrased: str, 
                                 neural_confidence: float, word_changes: int, 
                                 syntax_changes: int) -> Dict[str, float]:
        """
        Calculate comprehensive quality metrics
        
        Args:
            original: Original text
            paraphrased: Paraphrased text
            neural_confidence: Neural model confidence
            word_changes: Number of word changes
            syntax_changes: Number of syntax changes
            
        Returns:
            Dictionary of quality metrics
        """
        # Semantic similarity
        try:
            embeddings = self.semantic_model.encode([original, paraphrased])
            semantic_similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        except:
            semantic_similarity = 0.8  # Default fallback
        
        # Lexical diversity
        original_words = set(original.lower().split())
        paraphrased_words = set(paraphrased.lower().split())
        
        if len(original_words) > 0:
            lexical_diversity = len(paraphrased_words - original_words) / len(original_words)
        else:
            lexical_diversity = 0.0
        
        # Syntactic complexity (simplified)
        syntactic_complexity = min(1.0, syntax_changes / 3.0)
        
        # Fluency (based on length and structure similarity)
        length_ratio = len(paraphrased.split()) / max(len(original.split()), 1)
        fluency_score = 1.0 - abs(1.0 - length_ratio) * 0.5
        
        # Overall quality score
        quality_score = (
            semantic_similarity * 0.35 +
            lexical_diversity * 0.25 +
            syntactic_complexity * 0.20 +
            fluency_score * 0.20
        ) * 100
        
        return {
            "semantic_similarity": semantic_similarity,
            "lexical_diversity": lexical_diversity,
            "syntactic_complexity": syntactic_complexity,
            "fluency_score": fluency_score,
            "quality_score": quality_score
        }
    
    def paraphrase(self, text: str, method: str = "hybrid") -> IndoT5HybridResult:
        """
        Main paraphrasing method using hybrid approach
        
        Args:
            text: Input text to paraphrase
            method: Paraphrasing method ("hybrid", "neural", "rule-based")
            
        Returns:
            IndoT5HybridResult object
        """
        start_time = time.time()
        
        # Input validation
        if not text or not text.strip():
            return IndoT5HybridResult(
                original_text=text,
                paraphrased_text=text,
                method_used=method,
                transformations_applied=["Error: Empty input"],
                quality_score=0.0,
                confidence_score=0.0,
                neural_confidence=0.0,
                semantic_similarity=0.0,
                lexical_diversity=0.0,
                syntactic_complexity=0.0,
                fluency_score=0.0,
                processing_time=0.0,
                word_changes=0,
                syntax_changes=0,
                success=False,
                error_message="Empty input text"
            )
        
        # Check cache
        if self.enable_caching and text in self._result_cache:
            cached_result = self._result_cache[text]
            cached_result.processing_time = time.time() - start_time
            return cached_result
        
        try:
            transformations_applied = []
            
            if method == "hybrid":
                # Step 1: Neural paraphrase with IndoT5
                neural_result, neural_confidence = self._neural_paraphrase(text)
                transformations_applied.append("neural_generation")
                
                # Step 2: Rule-based enhancement
                if neural_confidence >= self.min_confidence:
                    # Apply synonym substitution
                    current_text, synonym_transforms, word_changes = self._apply_synonym_substitution(neural_result)
                    transformations_applied.extend(synonym_transforms)
                    
                    # Apply syntactic transformation
                    final_text, syntax_transforms, syntax_changes = self._apply_syntactic_transformation(current_text)
                    transformations_applied.extend(syntax_transforms)
                else:
                    # Low confidence, use original text for rule-based
                    current_text, synonym_transforms, word_changes = self._apply_synonym_substitution(text)
                    transformations_applied.extend(synonym_transforms)
                    
                    final_text, syntax_transforms, syntax_changes = self._apply_syntactic_transformation(current_text)
                    transformations_applied.extend(syntax_transforms)
            
            elif method == "neural":
                # Pure neural paraphrase
                final_text, neural_confidence = self._neural_paraphrase(text)
                transformations_applied.append("neural_generation")
                word_changes = len(set(text.lower().split()) - set(final_text.lower().split()))
                syntax_changes = 1 if final_text != text else 0
                
            elif method == "rule-based":
                # Pure rule-based paraphrase
                neural_confidence = 0.0
                
                # Apply synonym substitution
                current_text, synonym_transforms, word_changes = self._apply_synonym_substitution(text)
                transformations_applied.extend(synonym_transforms)
                
                # Apply syntactic transformation
                final_text, syntax_transforms, syntax_changes = self._apply_syntactic_transformation(current_text)
                transformations_applied.extend(syntax_transforms)
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(
                text, final_text, neural_confidence, word_changes, syntax_changes
            )
            
            # Create result
            result = IndoT5HybridResult(
                original_text=text,
                paraphrased_text=final_text,
                method_used=method,
                transformations_applied=transformations_applied,
                quality_score=quality_metrics["quality_score"],
                confidence_score=min(neural_confidence + quality_metrics["quality_score"] / 100, 1.0),
                neural_confidence=neural_confidence,
                semantic_similarity=quality_metrics["semantic_similarity"],
                lexical_diversity=quality_metrics["lexical_diversity"], 
                syntactic_complexity=quality_metrics["syntactic_complexity"],
                fluency_score=quality_metrics["fluency_score"],
                processing_time=time.time() - start_time,
                word_changes=word_changes,
                syntax_changes=syntax_changes,
                success=True
            )
            
            # Cache result
            if self.enable_caching:
                self._result_cache[text] = result
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Paraphrase failed: {e}")
            return IndoT5HybridResult(
                original_text=text,
                paraphrased_text=text,
                method_used=method,
                transformations_applied=["Error: Processing failed"],
                quality_score=0.0,
                confidence_score=0.0,
                neural_confidence=0.0,
                semantic_similarity=0.0,
                lexical_diversity=0.0,
                syntactic_complexity=0.0,
                fluency_score=0.0,
                processing_time=time.time() - start_time,
                word_changes=0,
                syntax_changes=0,
                success=False,
                error_message=str(e)
            )
    
    def generate_variations(self, text: str, num_variations: int = 3) -> List[IndoT5HybridResult]:
        """
        Generate multiple paraphrase variations
        
        Args:
            text: Input text
            num_variations: Number of variations to generate
            
        Returns:
            List of IndoT5HybridResult objects
        """
        variations = []
        
        for i in range(num_variations):
            # Vary the synonym rate and transformation parameters
            original_rate = self.synonym_rate
            original_transforms = self.max_transformations
            
            # Adjust parameters for variation
            self.synonym_rate = min(1.0, original_rate + (i * 0.1))
            self.max_transformations = min(5, original_transforms + i)
            
            # Generate variation
            result = self.paraphrase(text, method="hybrid")
            variations.append(result)
            
            # Restore original parameters
            self.synonym_rate = original_rate
            self.max_transformations = original_transforms
        
        # Sort by quality score
        variations.sort(key=lambda x: x.quality_score, reverse=True)
        
        return variations
    
    def batch_paraphrase(self, texts: List[str], method: str = "hybrid") -> List[IndoT5HybridResult]:
        """
        Process multiple texts in batch
        
        Args:
            texts: List of input texts
            method: Paraphrasing method
            
        Returns:
            List of IndoT5HybridResult objects
        """
        results = []
        
        for i, text in enumerate(texts):
            logger.info(f"Processing {i+1}/{len(texts)}: {text[:50]}...")
            result = self.paraphrase(text, method=method)
            results.append(result)
        
        return results
    
    def paraphrase_with_analysis(self, text: str) -> IndoT5HybridResult:
        """
        Paraphrase with detailed analysis
        
        Args:
            text: Input text
            
        Returns:
            IndoT5HybridResult with detailed metrics
        """
        return self.paraphrase(text, method="hybrid")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "use_gpu": self.use_gpu,
            "synonym_rate": self.synonym_rate,
            "min_confidence": self.min_confidence,
            "quality_threshold": self.quality_threshold,
            "max_transformations": self.max_transformations,
            "synonyms_loaded": len(self.synonym_data),
            "stopwords_loaded": len(self.stop_words)
        }

def create_indot5_hybrid_paraphraser(model_name: str = "Wikidepia/IndoT5-base", 
                                   **kwargs) -> IndoT5HybridParaphraser:
    """
    Factory function to create IndoT5 Hybrid Paraphraser
    
    Args:
        model_name: IndoT5 model name
        **kwargs: Additional parameters
        
    Returns:
        IndoT5HybridParaphraser instance
    """
    return IndoT5HybridParaphraser(model_name=model_name, **kwargs) 
