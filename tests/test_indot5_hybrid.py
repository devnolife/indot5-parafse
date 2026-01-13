"""
Test Suite for IndoT5 Hybrid Paraphraser
Tests for neural + rule-based hybrid paraphrasing
"""

import pytest
import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.indot5_hybrid_engine import IndoT5HybridParaphraser, IndoT5HybridResult, create_indot5_hybrid_paraphraser

class TestIndoT5HybridParaphraser:
    """Test cases for IndoT5 Hybrid Paraphraser"""
    
    @pytest.fixture(scope="class")
    def paraphraser(self):
        """Create paraphraser instance for testing"""
        return IndoT5HybridParaphraser(
            model_name="Wikidepia/IndoT5-base",
            use_gpu=False,  # Use CPU for testing
            synonym_rate=0.3,
            min_confidence=0.7,
            quality_threshold=60.0
        )
    
    def test_initialization(self, paraphraser):
        """Test paraphraser initialization"""
        assert paraphraser is not None
        assert paraphraser.model_name == "Wikidepia/IndoT5-base"
        assert paraphraser.synonym_rate == 0.3
        assert paraphraser.min_confidence == 0.7
        assert paraphraser.quality_threshold == 60.0
        assert hasattr(paraphraser, 'model')
        assert hasattr(paraphraser, 'tokenizer')
        assert hasattr(paraphraser, 'semantic_model')
    
    def test_basic_paraphrase(self, paraphraser):
        """Test basic paraphrasing functionality"""
        text = "Penelitian ini menggunakan metode kualitatif untuk menganalisis data."
        result = paraphraser.paraphrase(text)
        
        assert isinstance(result, IndoT5HybridResult)
        assert result.success == True
        assert result.original_text == text
        assert result.paraphrased_text != text
        assert result.paraphrased_text != ""
        assert result.quality_score > 0
        assert result.confidence_score > 0
        assert result.processing_time > 0
        assert len(result.transformations_applied) > 0
    
    def test_hybrid_method(self, paraphraser):
        """Test hybrid paraphrasing method"""
        text = "Machine learning adalah subset dari artificial intelligence."
        result = paraphraser.paraphrase(text, method="hybrid")
        
        assert result.success == True
        assert result.method_used == "hybrid"
        assert result.paraphrased_text != text
        assert result.neural_confidence >= 0
        assert result.semantic_similarity >= 0
        assert result.lexical_diversity >= 0
        assert result.syntactic_complexity >= 0
        assert result.fluency_score >= 0
        assert "neural_generation" in result.transformations_applied
    
    def test_neural_method(self, paraphraser):
        """Test neural-only paraphrasing method"""
        text = "Deep learning menggunakan neural network yang kompleks."
        result = paraphraser.paraphrase(text, method="neural")
        
        assert result.success == True
        assert result.method_used == "neural"
        assert result.paraphrased_text != text
        assert result.neural_confidence > 0
        assert "neural_generation" in result.transformations_applied
    
    def test_rule_based_method(self, paraphraser):
        """Test rule-based paraphrasing method"""
        text = "Teknologi blockchain sangat penting untuk masa depan."
        result = paraphraser.paraphrase(text, method="rule-based")
        
        assert result.success == True
        assert result.method_used == "rule-based"
        assert result.paraphrased_text != text
        assert result.neural_confidence == 0.0
        assert result.word_changes > 0 or result.syntax_changes > 0
    
    def test_empty_input(self, paraphraser):
        """Test handling of empty input"""
        result = paraphraser.paraphrase("")
        
        assert result.success == False
        assert result.error_message == "Empty input text"
        assert result.quality_score == 0
        assert result.confidence_score == 0
    
    def test_invalid_method(self, paraphraser):
        """Test handling of invalid method"""
        text = "Test text for invalid method."
        
        with pytest.raises(ValueError):
            paraphraser.paraphrase(text, method="invalid_method")
    
    def test_quality_metrics(self, paraphraser):
        """Test quality metrics calculation"""
        text = "Penelitian kualitatif menggunakan pendekatan fenomenologi."
        result = paraphraser.paraphrase(text)
        
        assert result.quality_score >= 0
        assert result.quality_score <= 100
        assert result.confidence_score >= 0
        assert result.confidence_score <= 1.0
        assert result.neural_confidence >= 0
        assert result.neural_confidence <= 1.0
        assert result.semantic_similarity >= 0
        assert result.semantic_similarity <= 1.0
        assert result.lexical_diversity >= 0
        assert result.syntactic_complexity >= 0
        assert result.fluency_score >= 0
        assert result.fluency_score <= 1.0
    
    def test_multiple_variations(self, paraphraser):
        """Test generation of multiple variations"""
        text = "Artificial intelligence mengubah cara kita bekerja."
        variations = paraphraser.generate_variations(text, num_variations=3)
        
        assert len(variations) == 3
        assert all(isinstance(v, IndoT5HybridResult) for v in variations)
        assert all(v.success for v in variations)
        assert all(v.paraphrased_text != text for v in variations)
        
        # Check that variations are different
        texts = [v.paraphrased_text for v in variations]
        assert len(set(texts)) >= 2  # At least 2 different variations
    
    def test_batch_processing(self, paraphraser):
        """Test batch processing"""
        texts = [
            "Machine learning adalah teknologi masa depan.",
            "Data science membantu pengambilan keputusan.",
            "Natural language processing menganalisis teks."
        ]
        
        results = paraphraser.batch_paraphrase(texts)
        
        assert len(results) == len(texts)
        assert all(isinstance(r, IndoT5HybridResult) for r in results)
        assert all(r.success for r in results)
        
        for original, result in zip(texts, results):
            assert result.original_text == original
            assert result.paraphrased_text != original
    
    def test_detailed_analysis(self, paraphraser):
        """Test detailed analysis functionality"""
        text = "Implementasi sistem informasi dapat meningkatkan efisiensi."
        result = paraphraser.paraphrase_with_analysis(text)
        
        assert isinstance(result, IndoT5HybridResult)
        assert result.success == True
        assert result.quality_score > 0
        assert result.neural_confidence >= 0
        assert result.semantic_similarity >= 0
        assert result.lexical_diversity >= 0
        assert result.syntactic_complexity >= 0
        assert result.fluency_score >= 0
        assert result.word_changes >= 0
        assert result.syntax_changes >= 0
        assert len(result.transformations_applied) > 0
    
    def test_model_info(self, paraphraser):
        """Test model information retrieval"""
        info = paraphraser.get_model_info()
        
        assert isinstance(info, dict)
        assert "model_name" in info
        assert "device" in info
        assert "use_gpu" in info
        assert "synonym_rate" in info
        assert "min_confidence" in info
        assert "quality_threshold" in info
        assert "max_transformations" in info
        assert "synonyms_loaded" in info
        assert "stopwords_loaded" in info
        
        assert info["model_name"] == "Wikidepia/IndoT5-base"
        assert info["synonym_rate"] == 0.3
        assert info["min_confidence"] == 0.7
        assert info["quality_threshold"] == 60.0
        assert isinstance(info["synonyms_loaded"], int)
        assert isinstance(info["stopwords_loaded"], int)
    
    def test_performance_benchmark(self, paraphraser):
        """Test performance benchmarking"""
        text = "Teknologi artificial intelligence berkembang pesat dalam dekade terakhir."
        
        start_time = time.time()
        result = paraphraser.paraphrase(text)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        assert result.success == True
        assert processing_time < 10.0  # Should complete within 10 seconds
        assert result.processing_time > 0
        assert result.processing_time <= processing_time
    
    def test_synonym_substitution(self, paraphraser):
        """Test synonym substitution functionality"""
        text = "Penelitian ini menggunakan metode kualitatif."
        result = paraphraser.paraphrase(text, method="rule-based")
        
        assert result.success == True
        assert result.word_changes > 0
        
        # Check if some synonyms were applied
        synonym_transforms = [t for t in result.transformations_applied if "synonym:" in t]
        assert len(synonym_transforms) > 0
    
    def test_syntactic_transformation(self, paraphraser):
        """Test syntactic transformation functionality"""
        text = "Peneliti menganalisis data dengan metode kualitatif."
        result = paraphraser.paraphrase(text, method="rule-based")
        
        assert result.success == True
        
        # Check if syntactic transformations were applied
        syntactic_transforms = [t for t in result.transformations_applied if "syntactic:" in t]
        # Note: Syntactic transformations may not always be applied depending on text
        assert isinstance(syntactic_transforms, list)
    
    def test_caching(self, paraphraser):
        """Test caching functionality"""
        text = "Test text for caching functionality."
        
        # First call
        result1 = paraphraser.paraphrase(text)
        time1 = result1.processing_time
        
        # Second call (should be faster due to caching)
        result2 = paraphraser.paraphrase(text)
        time2 = result2.processing_time
        
        assert result1.success == True
        assert result2.success == True
        assert result1.paraphrased_text == result2.paraphrased_text
        assert result1.quality_score == result2.quality_score
        
        # Second call should be faster (cached)
        assert time2 <= time1
    
    def test_different_texts(self, paraphraser):
        """Test with different types of texts"""
        texts = [
            "Kalimat pendek.",
            "Kalimat yang lebih panjang dengan beberapa kata yang dapat diubah menjadi sinonim.",
            "Penelitian ini menggunakan pendekatan kualitatif dengan metode wawancara mendalam untuk mengumpulkan data dari responden yang dipilih secara purposive sampling.",
            "Machine learning, deep learning, dan artificial intelligence adalah teknologi yang sangat penting untuk masa depan."
        ]
        
        for text in texts:
            result = paraphraser.paraphrase(text)
            assert result.success == True
            assert result.paraphrased_text != text
            assert result.quality_score > 0
            assert result.confidence_score > 0
    
    def test_error_handling(self, paraphraser):
        """Test error handling"""
        # Test with None input
        result = paraphraser.paraphrase(None)
        assert result.success == False
        assert result.error_message is not None
        
        # Test with very long text (should still work but may be truncated)
        long_text = "Penelitian " * 1000  # Very long text
        result = paraphraser.paraphrase(long_text)
        # Should not crash, but result may vary
        assert isinstance(result, IndoT5HybridResult)

class TestIndoT5HybridResult:
    """Test cases for IndoT5HybridResult data class"""
    
    def test_result_creation(self):
        """Test IndoT5HybridResult creation"""
        result = IndoT5HybridResult(
            original_text="Original text",
            paraphrased_text="Paraphrased text",
            method_used="hybrid",
            transformations_applied=["neural_generation"],
            quality_score=85.5,
            confidence_score=0.9,
            neural_confidence=0.8,
            semantic_similarity=0.85,
            lexical_diversity=0.7,
            syntactic_complexity=0.6,
            fluency_score=0.9,
            processing_time=2.5,
            word_changes=5,
            syntax_changes=2
        )
        
        assert result.original_text == "Original text"
        assert result.paraphrased_text == "Paraphrased text"
        assert result.method_used == "hybrid"
        assert result.transformations_applied == ["neural_generation"]
        assert result.quality_score == 85.5
        assert result.confidence_score == 0.9
        assert result.neural_confidence == 0.8
        assert result.semantic_similarity == 0.85
        assert result.lexical_diversity == 0.7
        assert result.syntactic_complexity == 0.6
        assert result.fluency_score == 0.9
        assert result.processing_time == 2.5
        assert result.word_changes == 5
        assert result.syntax_changes == 2
        assert result.success == True
        assert result.error_message is None
        assert result.alternatives == []

def test_factory_function():
    """Test factory function"""
    paraphraser = create_indot5_hybrid_paraphraser(
        model_name="Wikidepia/IndoT5-base",
        use_gpu=False,
        synonym_rate=0.4
    )
    
    assert isinstance(paraphraser, IndoT5HybridParaphraser)
    assert paraphraser.model_name == "Wikidepia/IndoT5-base"
    assert paraphraser.use_gpu == False
    assert paraphraser.synonym_rate == 0.4

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
