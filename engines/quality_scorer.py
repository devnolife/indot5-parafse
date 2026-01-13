"""
PlagiCheck - Quality Scorer Module
Advanced quality assessment for paraphrased text with multiple metrics
"""

import re
import math
import logging
from typing import Dict, List, Tuple, Optional
from nltk.tokenize import word_tokenize, sent_tokenize
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Container for quality assessment metrics"""
    lexical_diversity: float
    semantic_preservation: float
    syntactic_complexity: float
    readability_score: float
    fluency_score: float
    overall_score: float
    details: Dict

class QualityScorer:
    """
    Advanced quality assessment system for paraphrased text
    Evaluates multiple dimensions of paraphrase quality
    """
    
    def __init__(self, stopwords: set = None):
        """
        Initialize quality scorer
        
        Args:
            stopwords: Set of stopwords to exclude from analysis
        """
        self.stopwords = stopwords or set()
        
        # Quality weights for different metrics
        self.weights = {
            'lexical_diversity': 0.25,
            'semantic_preservation': 0.30,
            'syntactic_complexity': 0.20,
            'readability': 0.15,
            'fluency': 0.10
        }
        
        logger.info("✅ QualityScorer initialized")
    
    def calculate_comprehensive_score(self, original: str, paraphrased: str, 
                                    word_changes: int = 0, syntax_changes: int = 0) -> QualityMetrics:
        """
        Calculate comprehensive quality score with multiple metrics
        
        Args:
            original: Original text
            paraphrased: Paraphrased text  
            word_changes: Number of word substitutions made
            syntax_changes: Number of syntactic transformations made
            
        Returns:
            QualityMetrics object with detailed scores
        """
        try:
            # Calculate individual metrics
            lexical_score = self._calculate_lexical_diversity(original, paraphrased)
            semantic_score = self._calculate_semantic_preservation(original, paraphrased)
            syntactic_score = self._calculate_syntactic_complexity(original, paraphrased, syntax_changes)
            readability_score = self._calculate_readability(paraphrased)
            fluency_score = self._calculate_fluency(paraphrased)
            
            # Calculate weighted overall score
            overall_score = (
                lexical_score * self.weights['lexical_diversity'] +
                semantic_score * self.weights['semantic_preservation'] +
                syntactic_score * self.weights['syntactic_complexity'] +
                readability_score * self.weights['readability'] +
                fluency_score * self.weights['fluency']
            ) * 100
            
            # Prepare detailed metrics
            details = {
                'word_changes': word_changes,
                'syntax_changes': syntax_changes,
                'original_length': len(word_tokenize(original)),
                'paraphrased_length': len(word_tokenize(paraphrased)),
                'length_ratio': len(word_tokenize(paraphrased)) / max(len(word_tokenize(original)), 1),
                'character_ratio': len(paraphrased) / max(len(original), 1)
            }
            
            return QualityMetrics(
                lexical_diversity=lexical_score * 100,
                semantic_preservation=semantic_score * 100,
                syntactic_complexity=syntactic_score * 100,
                readability_score=readability_score * 100,
                fluency_score=fluency_score * 100,
                overall_score=min(100, max(0, overall_score)),
                details=details
            )
            
        except Exception as e:
            logger.error(f"❌ Error calculating quality score: {str(e)}")
            return QualityMetrics(
                lexical_diversity=0,
                semantic_preservation=0,
                syntactic_complexity=0,
                readability_score=0,
                fluency_score=0,
                overall_score=0,
                details={'error': str(e)}
            )
    
    def _calculate_lexical_diversity(self, original: str, paraphrased: str) -> float:
        """
        Calculate lexical diversity score (0-1)
        Measures how much vocabulary has changed
        """
        original_words = set(word_tokenize(original.lower())) - self.stopwords
        paraphrased_words = set(word_tokenize(paraphrased.lower())) - self.stopwords
        
        if not original_words:
            return 0.0
        
        # Calculate vocabulary overlap
        intersection = original_words.intersection(paraphrased_words)
        union = original_words.union(paraphrased_words)
        
        # Diversity = 1 - overlap ratio
        overlap_ratio = len(intersection) / len(union) if union else 1.0
        diversity = 1 - overlap_ratio
        
        # Normalize based on text length
        length_factor = min(len(paraphrased_words) / len(original_words), 1.5) if original_words else 0
        
        return diversity * length_factor
    
    def _calculate_semantic_preservation(self, original: str, paraphrased: str) -> float:
        """
        Calculate semantic preservation score (0-1)
        Measures how well meaning is preserved
        """
        # Simple approach using word overlap and structure
        original_words = word_tokenize(original.lower())
        paraphrased_words = word_tokenize(paraphrased.lower())
        
        # Remove stopwords for content comparison
        original_content = [w for w in original_words if w not in self.stopwords and w.isalpha()]
        paraphrased_content = [w for w in paraphrased_words if w not in self.stopwords and w.isalpha()]
        
        if not original_content:
            return 1.0
        
        # Calculate content word preservation
        original_set = set(original_content)
        paraphrased_set = set(paraphrased_content)
        
        # Some overlap is good for meaning preservation
        overlap = len(original_set.intersection(paraphrased_set))
        overlap_ratio = overlap / len(original_set)
        
        # Optimal range: 30-70% content word overlap
        if 0.3 <= overlap_ratio <= 0.7:
            preservation_score = 1.0
        elif overlap_ratio < 0.3:
            preservation_score = overlap_ratio / 0.3
        else:  # > 0.7
            preservation_score = (1.0 - overlap_ratio) / 0.3
        
        # Length similarity bonus
        length_ratio = len(paraphrased_content) / len(original_content)
        length_bonus = 1.0 - abs(1.0 - length_ratio) * 0.5
        
        return preservation_score * length_bonus
    
    def _calculate_syntactic_complexity(self, original: str, paraphrased: str, syntax_changes: int) -> float:
        """
        Calculate syntactic complexity score (0-1)
        Measures structural transformation quality
        """
        original_sentences = sent_tokenize(original)
        paraphrased_sentences = sent_tokenize(paraphrased)
        
        # Base score from syntax changes
        base_score = min(syntax_changes / 3.0, 1.0)  # Normalize to max 3 changes
        
        # Sentence structure analysis
        orig_avg_length = sum(len(word_tokenize(s)) for s in original_sentences) / len(original_sentences)
        para_avg_length = sum(len(word_tokenize(s)) for s in paraphrased_sentences) / len(paraphrased_sentences)
        
        # Variance in sentence length (complexity indicator)
        orig_lengths = [len(word_tokenize(s)) for s in original_sentences]
        para_lengths = [len(word_tokenize(s)) for s in paraphrased_sentences]
        
        orig_variance = self._calculate_variance(orig_lengths)
        para_variance = self._calculate_variance(para_lengths)
        
        # Structure change bonus
        structure_change = abs(para_variance - orig_variance) / max(orig_variance, 1)
        structure_bonus = min(structure_change, 0.5)
        
        # Length change consideration
        length_change = abs(para_avg_length - orig_avg_length) / max(orig_avg_length, 1)
        length_bonus = min(length_change * 0.3, 0.3)
        
        return min(base_score + structure_bonus + length_bonus, 1.0)
    
    def _calculate_readability(self, text: str) -> float:
        """
        Calculate readability score (0-1)
        Based on sentence length and complexity
        """
        sentences = sent_tokenize(text)
        if not sentences:
            return 0.0
        
        words = word_tokenize(text)
        
        # Average sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Optimal range: 12-20 words per sentence for Indonesian
        if 12 <= avg_sentence_length <= 20:
            length_score = 1.0
        elif avg_sentence_length < 12:
            length_score = avg_sentence_length / 12
        else:  # > 20
            length_score = max(0, 1.0 - (avg_sentence_length - 20) / 20)
        
        # Complexity indicators
        # Count complex punctuation
        complex_punct = text.count(';') + text.count(':') + text.count('—')
        complexity_penalty = min(complex_punct * 0.1, 0.3)
        
        # Word length variety
        word_lengths = [len(w) for w in words if w.isalpha()]
        if word_lengths:
            avg_word_length = sum(word_lengths) / len(word_lengths)
            word_length_score = 1.0 if avg_word_length <= 6 else max(0, 1.0 - (avg_word_length - 6) / 6)
        else:
            word_length_score = 0.5
        
        return max(0, (length_score + word_length_score) / 2 - complexity_penalty)
    
    def _calculate_fluency(self, text: str) -> float:
        """
        Calculate fluency score (0-1)
        Based on grammatical patterns and flow
        """
        # Simple fluency indicators
        
        # 1. Proper capitalization
        sentences = sent_tokenize(text)
        properly_capitalized = sum(1 for s in sentences if s and s[0].isupper())
        capitalization_score = properly_capitalized / len(sentences) if sentences else 0
        
        # 2. Punctuation consistency
        punctuation_score = 1.0
        if text.count('..') > 0 or text.count('??') > 0 or text.count('!!') > 1:
            punctuation_score -= 0.2
        
        # 3. Word repetition (penalty for too much repetition)
        words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
        if words:
            unique_words = len(set(words))
            repetition_score = unique_words / len(words)
        else:
            repetition_score = 1.0
        
        # 4. Connector word usage (good for flow)
        connectors = ['dan', 'atau', 'tetapi', 'namun', 'karena', 'sehingga', 'jika', 'ketika']
        connector_count = sum(1 for word in words if word in connectors)
        connector_ratio = connector_count / len(words) if words else 0
        connector_score = min(connector_ratio * 10, 1.0)  # Reasonable amount of connectors
        
        # 5. Sentence flow (no too short or too long sentences)
        sentence_lengths = [len(word_tokenize(s)) for s in sentences]
        flow_penalty = 0
        for length in sentence_lengths:
            if length < 4 or length > 40:
                flow_penalty += 0.1
        
        flow_score = max(0, 1.0 - flow_penalty)
        
        # Combine scores
        fluency = (
            capitalization_score * 0.2 +
            punctuation_score * 0.2 +
            repetition_score * 0.3 +
            connector_score * 0.1 +
            flow_score * 0.2
        )
        
        return fluency
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values"""
        if not values:
            return 0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def get_quality_category(self, score: float) -> str:
        """
        Get quality category based on score
        
        Args:
            score: Quality score (0-100)
            
        Returns:
            Quality category string
        """
        if score >= 85:
            return "Excellent"
        elif score >= 70:
            return "Good"
        elif score >= 55:
            return "Fair"
        elif score >= 40:
            return "Poor"
        else:
            return "Very Poor"
    
    def get_improvement_suggestions(self, metrics: QualityMetrics) -> List[str]:
        """
        Get suggestions for improving paraphrase quality
        
        Args:
            metrics: QualityMetrics object
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        if metrics.lexical_diversity < 60:
            suggestions.append("Increase vocabulary variation by using more synonyms")
        
        if metrics.semantic_preservation < 70:
            suggestions.append("Ensure meaning preservation by maintaining key concepts")
        
        if metrics.syntactic_complexity < 50:
            suggestions.append("Add more structural transformations (voice changes, clause reordering)")
        
        if metrics.readability_score < 60:
            suggestions.append("Improve readability by adjusting sentence length and complexity")
        
        if metrics.fluency_score < 70:
            suggestions.append("Enhance fluency by checking grammar and sentence flow")
        
        # Length-based suggestions
        if 'length_ratio' in metrics.details:
            ratio = metrics.details['length_ratio']
            if ratio < 0.7:
                suggestions.append("Paraphrase is too short - consider expanding ideas")
            elif ratio > 1.5:
                suggestions.append("Paraphrase is too long - consider condensing")
        
        return suggestions
