"""
Text Validator for Paraphrasing
Validation utilities for input/output text quality and correctness
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from nltk.tokenize import word_tokenize, sent_tokenize
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass 
class ValidationResult:
    """Container for validation results"""
    is_valid: bool
    score: float
    issues: List[str]
    warnings: List[str]
    suggestions: List[str]
    metadata: Dict

class TextValidator:
    """
    Comprehensive text validation for paraphrasing
    Validates input quality, output correctness, and semantic preservation
    """
    
    def __init__(self, min_length: int = 10, max_length: int = 10000):
        """
        Initialize text validator
        
        Args:
            min_length: Minimum text length in characters
            max_length: Maximum text length in characters
        """
        self.min_length = min_length
        self.max_length = max_length
        
        # Indonesian stopwords for analysis
        self.stopwords = {
            'dan', 'atau', 'yang', 'adalah', 'dengan', 'untuk', 'dalam', 'dari',
            'pada', 'ke', 'di', 'ini', 'itu', 'akan', 'dapat', 'juga', 'tidak',
            'ada', 'sudah', 'telah', 'harus', 'bisa', 'lebih', 'sangat'
        }
        
        # Common problematic patterns
        self.problematic_patterns = [
            (r'(.)\1{3,}', "Excessive character repetition"),
            (r'\b(\w+)\s+\1\b', "Word repetition"),
            (r'[.]{4,}', "Excessive dots"),
            (r'[!]{2,}', "Excessive exclamation marks"),
            (r'[?]{2,}', "Excessive question marks"),
            (r'\s{3,}', "Excessive whitespace"),
        ]
        
        logger.info("✅ TextValidator initialized")
    
    def validate_input_text(self, text: str) -> ValidationResult:
        """
        Validate input text for paraphrasing
        
        Args:
            text: Input text to validate
            
        Returns:
            ValidationResult object
        """
        issues = []
        warnings = []
        suggestions = []
        metadata = {}
        
        # Basic validation
        if not text or not text.strip():
            return ValidationResult(
                is_valid=False,
                score=0.0,
                issues=["Text is empty or contains only whitespace"],
                warnings=[],
                suggestions=["Provide non-empty text for paraphrasing"],
                metadata={}
            )
        
        text = text.strip()
        metadata['original_length'] = len(text)
        metadata['word_count'] = len(word_tokenize(text))
        metadata['sentence_count'] = len(sent_tokenize(text))
        
        # Length validation
        if len(text) < self.min_length:
            issues.append(f"Text too short (minimum {self.min_length} characters)")
        elif len(text) > self.max_length:
            issues.append(f"Text too long (maximum {self.max_length} characters)")
        
        # Content validation
        self._validate_content_quality(text, issues, warnings, suggestions)
        
        # Structure validation
        self._validate_text_structure(text, issues, warnings, suggestions)
        
        # Language validation
        self._validate_language_quality(text, issues, warnings, suggestions)
        
        # Calculate validation score
        score = self._calculate_validation_score(issues, warnings)
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            score=score,
            issues=issues,
            warnings=warnings,
            suggestions=suggestions,
            metadata=metadata
        )
    
    def validate_paraphrase_output(self, original: str, paraphrased: str) -> ValidationResult:
        """
        Validate paraphrased output quality
        
        Args:
            original: Original text
            paraphrased: Paraphrased text
            
        Returns:
            ValidationResult object
        """
        issues = []
        warnings = []
        suggestions = []
        metadata = {}
        
        # Basic validation
        if not paraphrased or not paraphrased.strip():
            return ValidationResult(
                is_valid=False,
                score=0.0,
                issues=["Paraphrased text is empty"],
                warnings=[],
                suggestions=["Ensure paraphrasing produces valid output"],
                metadata={}
            )
        
        # Metadata
        metadata.update({
            'original_length': len(original),
            'paraphrased_length': len(paraphrased),
            'length_ratio': len(paraphrased) / len(original) if original else 0,
            'original_words': len(word_tokenize(original)),
            'paraphrased_words': len(word_tokenize(paraphrased)),
        })
        
        # Similarity validation
        self._validate_similarity(original, paraphrased, issues, warnings, suggestions)
        
        # Quality validation
        self._validate_paraphrase_quality(original, paraphrased, issues, warnings, suggestions)
        
        # Coherence validation
        self._validate_coherence(paraphrased, issues, warnings, suggestions)
        
        # Calculate score
        score = self._calculate_validation_score(issues, warnings)
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            score=score,
            issues=issues,
            warnings=warnings,
            suggestions=suggestions,
            metadata=metadata
        )
    
    def _validate_content_quality(self, text: str, issues: List[str], 
                                warnings: List[str], suggestions: List[str]):
        """Validate content quality"""
        
        # Check for problematic patterns
        for pattern, description in self.problematic_patterns:
            if re.search(pattern, text):
                warnings.append(description)
        
        # Check character composition
        alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
        if alpha_ratio < 0.5:
            warnings.append("Low alphabetic character ratio")
            suggestions.append("Ensure text contains sufficient alphabetic content")
        
        # Check for meaningful content
        words = word_tokenize(text.lower())
        content_words = [w for w in words if w.isalpha() and w not in self.stopwords]
        
        if len(content_words) < 3:
            issues.append("Insufficient content words")
            suggestions.append("Provide text with more meaningful content")
        
        # Check sentence structure
        sentences = sent_tokenize(text)
        if len(sentences) == 0:
            issues.append("No proper sentences detected")
        elif len(sentences) == 1 and len(words) > 50:
            warnings.append("Very long single sentence")
            suggestions.append("Consider breaking into shorter sentences")
    
    def _validate_text_structure(self, text: str, issues: List[str],
                                warnings: List[str], suggestions: List[str]):
        """Validate text structure"""
        
        # Check sentence endings
        sentences = sent_tokenize(text)
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            # Check if sentence ends properly
            if not sentence.rstrip()[-1] in '.!?':
                warnings.append(f"Sentence {i+1} doesn't end with proper punctuation")
        
        # Check capitalization
        for i, sentence in enumerate(sentences):
            if sentence and not sentence[0].isupper():
                warnings.append(f"Sentence {i+1} doesn't start with capital letter")
        
        # Check paragraph structure
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            for i, para in enumerate(paragraphs):
                if len(para.strip()) < 50:
                    warnings.append(f"Paragraph {i+1} is very short")
    
    def _validate_language_quality(self, text: str, issues: List[str],
                                 warnings: List[str], suggestions: List[str]):
        """Validate language quality for Indonesian"""
        
        # Check for common Indonesian patterns
        words = word_tokenize(text.lower())
        
        # Check for reasonable Indonesian word patterns
        indonesian_patterns = [
            r'ber\w+',  # ber- prefix
            r'me\w+',   # me- prefix  
            r'di\w+',   # di- prefix
            r'ke\w+',   # ke- prefix
            r'\w+an$',  # -an suffix
            r'\w+kan$', # -kan suffix
        ]
        
        pattern_matches = 0
        for word in words:
            if any(re.match(pattern, word) for pattern in indonesian_patterns):
                pattern_matches += 1
        
        if words and pattern_matches / len(words) < 0.1:
            warnings.append("Text may not be Indonesian or has unusual patterns")
        
        # Check for balanced sentence complexity
        sentences = sent_tokenize(text)
        if sentences:
            avg_length = sum(len(word_tokenize(s)) for s in sentences) / len(sentences)
            if avg_length < 5:
                warnings.append("Sentences are very short")
                suggestions.append("Consider adding more detail to sentences")
            elif avg_length > 30:
                warnings.append("Sentences are very long")
                suggestions.append("Consider breaking long sentences")
    
    def _validate_similarity(self, original: str, paraphrased: str,
                           issues: List[str], warnings: List[str], suggestions: List[str]):
        """Validate similarity between original and paraphrased text"""
        
        # Check if texts are identical
        if original.strip().lower() == paraphrased.strip().lower():
            issues.append("Paraphrased text is identical to original")
            suggestions.append("Ensure paraphrasing produces different text")
            return
        
        # Calculate word overlap
        orig_words = set(word_tokenize(original.lower()))
        para_words = set(word_tokenize(paraphrased.lower()))
        
        # Remove stopwords for content comparison
        orig_content = orig_words - self.stopwords
        para_content = para_words - self.stopwords
        
        if orig_content and para_content:
            overlap = len(orig_content.intersection(para_content))
            overlap_ratio = overlap / len(orig_content)
            
            if overlap_ratio > 0.8:
                warnings.append("Very high word overlap with original")
                suggestions.append("Try using more synonyms and restructuring")
            elif overlap_ratio < 0.2:
                warnings.append("Very low word overlap - meaning may be lost")
                suggestions.append("Ensure key concepts are preserved")
        
        # Length comparison
        length_ratio = len(paraphrased) / len(original) if original else 0
        if length_ratio < 0.5:
            warnings.append("Paraphrased text is much shorter than original")
        elif length_ratio > 2.0:
            warnings.append("Paraphrased text is much longer than original")
    
    def _validate_paraphrase_quality(self, original: str, paraphrased: str,
                                   issues: List[str], warnings: List[str], suggestions: List[str]):
        """Validate paraphrase quality"""
        
        # Check for proper transformations
        orig_sentences = sent_tokenize(original)
        para_sentences = sent_tokenize(paraphrased)
        
        # Number of sentences should be similar
        if abs(len(orig_sentences) - len(para_sentences)) > 2:
            warnings.append("Significant difference in sentence count")
        
        # Check for structural changes
        orig_structure = self._analyze_structure(original)
        para_structure = self._analyze_structure(paraphrased)
        
        structure_changes = 0
        for key in orig_structure:
            if key in para_structure:
                if abs(orig_structure[key] - para_structure[key]) > 0.1:
                    structure_changes += 1
        
        if structure_changes == 0:
            warnings.append("No significant structural changes detected")
            suggestions.append("Try applying more syntactic transformations")
    
    def _validate_coherence(self, text: str, issues: List[str],
                          warnings: List[str], suggestions: List[str]):
        """Validate text coherence and flow"""
        
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return  # Can't validate coherence for single sentence
        
        # Check for logical connectors
        connectors = ['dan', 'tetapi', 'namun', 'karena', 'sehingga', 'jika', 'ketika']
        connector_count = 0
        
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            if any(conn in words for conn in connectors):
                connector_count += 1
        
        if connector_count == 0 and len(sentences) > 2:
            warnings.append("No logical connectors found - may affect coherence")
            suggestions.append("Consider adding connecting words between ideas")
        
        # Check sentence transitions
        for i in range(len(sentences) - 1):
            curr_words = set(word_tokenize(sentences[i].lower()))
            next_words = set(word_tokenize(sentences[i + 1].lower()))
            
            # Remove stopwords
            curr_content = curr_words - self.stopwords
            next_content = next_words - self.stopwords
            
            if curr_content and next_content:
                overlap = len(curr_content.intersection(next_content))
                if overlap == 0:
                    warnings.append(f"No content overlap between sentences {i+1} and {i+2}")
    
    def _analyze_structure(self, text: str) -> Dict[str, float]:
        """Analyze text structure for comparison"""
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        
        if not words:
            return {}
        
        return {
            'avg_word_length': sum(len(w) for w in words if w.isalpha()) / len([w for w in words if w.isalpha()]),
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'punctuation_ratio': sum(1 for c in text if c in '.,!?;:') / len(text),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text),
        }
    
    def _calculate_validation_score(self, issues: List[str], warnings: List[str]) -> float:
        """Calculate validation score based on issues and warnings"""
        if issues:
            return 0.0  # Any issues result in invalid
        
        # Start with perfect score
        score = 100.0
        
        # Deduct points for warnings
        score -= len(warnings) * 10
        
        # Ensure score is within bounds
        return max(0.0, min(100.0, score))
    
    def validate_batch_texts(self, texts: List[str]) -> List[ValidationResult]:
        """
        Validate multiple texts in batch
        
        Args:
            texts: List of texts to validate
            
        Returns:
            List of ValidationResult objects
        """
        results = []
        
        for i, text in enumerate(texts):
            try:
                result = self.validate_input_text(text)
                results.append(result)
            except Exception as e:
                logger.error(f"❌ Error validating text {i+1}: {str(e)}")
                results.append(ValidationResult(
                    is_valid=False,
                    score=0.0,
                    issues=[f"Validation error: {str(e)}"],
                    warnings=[],
                    suggestions=["Check text format and content"],
                    metadata={'error': True}
                ))
        
        return results
    
    def get_validation_summary(self, results: List[ValidationResult]) -> Dict:
        """
        Get summary of validation results
        
        Args:
            results: List of ValidationResult objects
            
        Returns:
            Summary dictionary
        """
        if not results:
            return {}
        
        valid_count = sum(1 for r in results if r.is_valid)
        avg_score = sum(r.score for r in results) / len(results)
        
        total_issues = sum(len(r.issues) for r in results)
        total_warnings = sum(len(r.warnings) for r in results)
        
        # Most common issues
        all_issues = []
        all_warnings = []
        
        for result in results:
            all_issues.extend(result.issues)
            all_warnings.extend(result.warnings)
        
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        warning_counts = {}
        for warning in all_warnings:
            warning_counts[warning] = warning_counts.get(warning, 0) + 1
        
        return {
            'total_texts': len(results),
            'valid_texts': valid_count,
            'invalid_texts': len(results) - valid_count,
            'average_score': round(avg_score, 2),
            'total_issues': total_issues,
            'total_warnings': total_warnings,
            'common_issues': sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            'common_warnings': sorted(warning_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }
