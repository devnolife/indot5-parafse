"""
Text Processing Utilities for Paraphrasing
Enhanced text preprocessing and normalization
"""

import re
import unicodedata
import logging
from typing import List, Dict, Tuple, Optional
from nltk.tokenize import word_tokenize, sent_tokenize

logger = logging.getLogger(__name__)

class TextProcessor:
    """
    Advanced text processing utilities for paraphrasing
    Handles normalization, cleaning, and preprocessing
    """
    
    def __init__(self):
        """Initialize text processor"""
        # Common Indonesian abbreviations and their expansions
        self.abbreviations = {
            'yg': 'yang',
            'dg': 'dengan', 
            'dr': 'dari',
            'ke': 'ke',
            'pd': 'pada',
            'tdk': 'tidak',
            'hrs': 'harus',
            'utk': 'untuk',
            'dlm': 'dalam',
            'krn': 'karena',
            'dgn': 'dengan',
            'tsb': 'tersebut',
            'dll': 'dan lain-lain',
            'dsb': 'dan sebagainya'
        }
        
        # Technical terms to preserve
        self.technical_terms = {
            'API', 'HTTP', 'JSON', 'XML', 'SQL', 'HTML', 'CSS', 'JavaScript',
            'Python', 'Java', 'C++', 'PHP', 'ML', 'AI', 'IoT', 'VR', 'AR',
            'GPS', 'USB', 'CPU', 'GPU', 'RAM', 'SSD', 'HDD', 'OS', 'UI', 'UX'
        }
        
        logger.info("✅ TextProcessor initialized")
    
    def normalize_text(self, text: str, preserve_technical: bool = True) -> str:
        """
        Comprehensive text normalization
        
        Args:
            text: Input text to normalize
            preserve_technical: Whether to preserve technical terms
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Store technical terms if preserving
        preserved_terms = {}
        if preserve_technical:
            text, preserved_terms = self._preserve_technical_terms(text)
        
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Fix common encoding issues
        text = self._fix_encoding_issues(text)
        
        # Expand abbreviations
        text = self._expand_abbreviations(text)
        
        # Normalize whitespace
        text = self._normalize_whitespace(text)
        
        # Normalize punctuation
        text = self._normalize_punctuation(text)
        
        # Restore technical terms
        if preserve_technical and preserved_terms:
            text = self._restore_technical_terms(text, preserved_terms)
        
        return text.strip()
    
    def _preserve_technical_terms(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Preserve technical terms by replacing with placeholders"""
        preserved = {}
        
        for i, term in enumerate(self.technical_terms):
            if term in text:
                placeholder = f"__TECH_TERM_{i}__"
                preserved[placeholder] = term
                text = text.replace(term, placeholder)
        
        return text, preserved
    
    def _restore_technical_terms(self, text: str, preserved_terms: Dict[str, str]) -> str:
        """Restore preserved technical terms"""
        for placeholder, term in preserved_terms.items():
            text = text.replace(placeholder, term)
        return text
    
    def _fix_encoding_issues(self, text: str) -> str:
        """Fix common encoding and character issues"""
        # Fix smart quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Fix dashes
        text = text.replace('–', '-').replace('—', '-')
        
        # Fix ellipsis
        text = text.replace('…', '...')
        
        # Remove invisible characters
        text = re.sub(r'[\u200b-\u200d\ufeff]', '', text)
        
        return text
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common Indonesian abbreviations"""
        for abbrev, expansion in self.abbreviations.items():
            # Word boundary replacement
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize all whitespace characters"""
        # Replace tabs and multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace from lines
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)
        
        # Remove excessive line breaks
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text
    
    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation spacing and usage"""
        # Fix spacing around punctuation
        text = re.sub(r'\s*([,.!?;:])\s*', r'\1 ', text)
        text = re.sub(r'\s*([,.!?;:])$', r'\1', text, flags=re.MULTILINE)
        
        # Fix quotation marks
        text = re.sub(r'\s*"\s*', '"', text)
        text = re.sub(r'\s*"\s*', '"', text)
        
        # Fix parentheses
        text = re.sub(r'\s*\(\s*', ' (', text)
        text = re.sub(r'\s*\)\s*', ') ', text)
        
        # Remove space before punctuation at end of word
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        
        # Ensure single space after punctuation
        text = re.sub(r'([,.!?;:])\s*', r'\1 ', text)
        
        return text
    
    def preprocess_for_paraphrasing(self, text: str) -> str:
        """
        Specialized preprocessing for paraphrasing
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text ready for paraphrasing
        """
        if not text:
            return ""
        
        # Basic normalization
        text = self.normalize_text(text)
        
        # Additional preprocessing for paraphrasing
        
        # Ensure proper sentence separation
        text = self._ensure_sentence_separation(text)
        
        # Handle special cases
        text = self._handle_special_cases(text)
        
        return text
    
    def _ensure_sentence_separation(self, text: str) -> str:
        """Ensure proper sentence separation"""
        # Add space after sentence-ending punctuation if missing
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        
        # Handle common sentence boundary issues
        text = re.sub(r'([.!?])\s*([a-z])', lambda m: m.group(1) + ' ' + m.group(2).upper(), text)
        
        return text
    
    def _handle_special_cases(self, text: str) -> str:
        """Handle special cases in Indonesian text"""
        # Handle common Indonesian patterns
        
        # Fix "di-" prefix spacing
        text = re.sub(r'\bdi\s+([a-z])', r'di\1', text)
        
        # Fix "ke-" prefix spacing  
        text = re.sub(r'\bke\s+([a-z])', r'ke\1', text)
        
        # Handle number formatting
        text = re.sub(r'(\d+)\s*,\s*(\d+)', r'\1,\2', text)
        
        return text
    
    def postprocess_paraphrased_text(self, text: str) -> str:
        """
        Post-process paraphrased text for final output
        
        Args:
            text: Paraphrased text to post-process
            
        Returns:
            Final processed text
        """
        if not text:
            return ""
        
        # Fix capitalization
        text = self._fix_capitalization(text)
        
        # Clean up spacing
        text = self._final_spacing_cleanup(text)
        
        # Validate and fix punctuation
        text = self._final_punctuation_check(text)
        
        return text.strip()
    
    def _fix_capitalization(self, text: str) -> str:
        """Fix capitalization issues"""
        sentences = sent_tokenize(text)
        fixed_sentences = []
        
        for sentence in sentences:
            if sentence:
                # Capitalize first letter of sentence
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                fixed_sentences.append(sentence)
        
        return ' '.join(fixed_sentences)
    
    def _final_spacing_cleanup(self, text: str) -> str:
        """Final cleanup of spacing issues"""
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix punctuation spacing
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])\s*', r'\1 ', text)
        
        # Clean up end of text
        text = re.sub(r'\s+$', '', text)
        
        return text
    
    def _final_punctuation_check(self, text: str) -> str:
        """Final punctuation validation"""
        # Ensure text ends with proper punctuation
        if text and not text[-1] in '.!?':
            text += '.'
        
        # Remove duplicate punctuation
        text = re.sub(r'([.!?])+', r'\1', text)
        
        return text
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract and clean sentences from text
        
        Args:
            text: Input text
            
        Returns:
            List of cleaned sentences
        """
        if not text:
            return []
        
        # Normalize text first
        text = self.normalize_text(text)
        
        # Extract sentences
        sentences = sent_tokenize(text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 3:  # Filter very short sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def extract_keywords(self, text: str, min_length: int = 3) -> List[str]:
        """
        Extract keywords from text
        
        Args:
            text: Input text
            min_length: Minimum keyword length
            
        Returns:
            List of keywords
        """
        if not text:
            return []
        
        # Normalize text
        text = self.normalize_text(text)
        
        # Tokenize
        words = word_tokenize(text.lower())
        
        # Filter keywords
        keywords = []
        for word in words:
            if (word.isalpha() and 
                len(word) >= min_length and
                word not in {'dan', 'atau', 'yang', 'adalah', 'dengan', 'untuk', 'dalam', 'dari'}):
                keywords.append(word)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords
    
    def calculate_text_statistics(self, text: str) -> Dict:
        """
        Calculate comprehensive text statistics
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with text statistics
        """
        if not text:
            return {}
        
        # Basic counts
        characters = len(text)
        characters_no_spaces = len(text.replace(' ', ''))
        words = word_tokenize(text)
        sentences = sent_tokenize(text)
        
        # Advanced statistics
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # Vocabulary diversity
        unique_words = len(set(word.lower() for word in words if word.isalpha()))
        vocabulary_diversity = unique_words / len(words) if words else 0
        
        return {
            'characters': characters,
            'characters_no_spaces': characters_no_spaces,
            'words': len(words),
            'sentences': len(sentences),
            'paragraphs': len([p for p in text.split('\n\n') if p.strip()]),
            'avg_word_length': round(avg_word_length, 2),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'unique_words': unique_words,
            'vocabulary_diversity': round(vocabulary_diversity, 3),
        }
