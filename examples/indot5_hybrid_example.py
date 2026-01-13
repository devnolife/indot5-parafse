"""
IndoT5 Hybrid Paraphraser Example
Demonstrates usage of IndoT5 neural processing with rule-based transformations
"""

import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.indot5_hybrid_engine import IndoT5HybridParaphraser

def basic_example():
    """Basic IndoT5 hybrid paraphrasing example"""
    print("=" * 60)
    print("INDOT5 HYBRID PARAPHRASER - BASIC EXAMPLE")
    print("=" * 60)
    
    # Initialize paraphraser
    print("🚀 Initializing IndoT5 Hybrid Paraphraser...")
    paraphraser = IndoT5HybridParaphraser(
        model_name="Wikidepia/IndoT5-base",
        use_gpu=True,
        synonym_rate=0.3,
        min_confidence=0.7,
        quality_threshold=75.0
    )
    
    print("✅ Paraphraser initialized successfully!")
    print()
    
    # Example text
    text = "Penelitian ini menggunakan metode kualitatif untuk menganalisis data wawancara dari responden."
    
    print(f"📝 Original Text:")
    print(f"   '{text}'")
    print()
    
    # Generate paraphrase
    print("🔄 Generating paraphrase...")
    result = paraphraser.paraphrase(text)
    
    # Display results
    print("📊 RESULTS:")
    print(f"   ✅ Success: {result.success}")
    print(f"   📝 Paraphrased: '{result.paraphrased_text}'")
    print(f"   🎯 Method Used: {result.method_used}")
    print(f"   📊 Quality Score: {result.quality_score:.1f}")
    print(f"   🔥 Confidence: {result.confidence_score:.3f}")
    print(f"   🧠 Neural Confidence: {result.neural_confidence:.3f}")
    print(f"   🔗 Semantic Similarity: {result.semantic_similarity:.3f}")
    print(f"   📈 Lexical Diversity: {result.lexical_diversity:.3f}")
    print(f"   🔧 Syntactic Complexity: {result.syntactic_complexity:.3f}")
    print(f"   ✨ Fluency Score: {result.fluency_score:.3f}")
    print(f"   ⏱️ Processing Time: {result.processing_time:.2f}s")
    print(f"   🔄 Word Changes: {result.word_changes}")
    print(f"   🏗️ Syntax Changes: {result.syntax_changes}")
    
    print()
    print("🔧 TRANSFORMATIONS APPLIED:")
    for i, transform in enumerate(result.transformations_applied, 1):
        print(f"   {i}. {transform}")
    
    print()

def method_comparison_example():
    """Compare different paraphrasing methods"""
    print("=" * 60)
    print("METHOD COMPARISON EXAMPLE")
    print("=" * 60)
    
    # Initialize paraphraser
    paraphraser = IndoT5HybridParaphraser()
    
    # Test text
    text = "Teknologi artificial intelligence berkembang pesat dalam dekade terakhir dan memberikan dampak signifikan."
    
    print(f"📝 Test Text:")
    print(f"   '{text}'")
    print()
    
    # Test different methods
    methods = ["hybrid", "neural", "rule-based"]
    
    for method in methods:
        print(f"🔧 Method: {method.upper()}")
        
        start_time = time.time()
        result = paraphraser.paraphrase(text, method=method)
        processing_time = time.time() - start_time
        
        if result.success:
            print(f"   ✅ Success!")
            print(f"   📝 Result: '{result.paraphrased_text}'")
            print(f"   📊 Quality: {result.quality_score:.1f}")
            print(f"   🔥 Confidence: {result.confidence_score:.3f}")
            print(f"   ⏱️ Time: {processing_time:.2f}s")
            print(f"   🔄 Changes: {result.word_changes} words, {result.syntax_changes} syntax")
        else:
            print(f"   ❌ Failed: {result.error_message}")
        
        print()

def multiple_variations_example():
    """Generate multiple paraphrase variations"""
    print("=" * 60)
    print("MULTIPLE VARIATIONS EXAMPLE")
    print("=" * 60)
    
    # Initialize paraphraser
    paraphraser = IndoT5HybridParaphraser()
    
    # Test text
    text = "Penelitian kualitatif menggunakan pendekatan fenomenologi untuk memahami pengalaman subjektif partisipan."
    
    print(f"📝 Original Text:")
    print(f"   '{text}'")
    print()
    
    # Generate multiple variations
    print("🎲 Generating 3 variations...")
    variations = paraphraser.generate_variations(text, num_variations=3)
    
    for i, variation in enumerate(variations, 1):
        print(f"📝 Variation {i} (Quality: {variation.quality_score:.1f}):")
        print(f"   '{variation.paraphrased_text}'")
        print(f"   🔧 Transformations: {len(variation.transformations_applied)}")
        print()

def batch_processing_example():
    """Process multiple texts in batch"""
    print("=" * 60)
    print("BATCH PROCESSING EXAMPLE")
    print("=" * 60)
    
    # Initialize paraphraser
    paraphraser = IndoT5HybridParaphraser()
    
    # Multiple texts
    texts = [
        "Machine learning adalah subset dari artificial intelligence.",
        "Deep learning menggunakan neural network yang kompleks.",
        "Natural language processing membantu komputer memahami bahasa manusia."
    ]
    
    print(f"📝 Processing {len(texts)} texts...")
    print()
    
    # Process batch
    results = paraphraser.batch_paraphrase(texts)
    
    # Display results
    for i, (original, result) in enumerate(zip(texts, results), 1):
        print(f"📝 Text {i}:")
        print(f"   Original: '{original}'")
        print(f"   Paraphrased: '{result.paraphrased_text}'")
        print(f"   Quality: {result.quality_score:.1f}")
        print(f"   Success: {result.success}")
        print()

def detailed_analysis_example():
    """Detailed paraphrase analysis example"""
    print("=" * 60)
    print("DETAILED ANALYSIS EXAMPLE")
    print("=" * 60)
    
    # Initialize paraphraser
    paraphraser = IndoT5HybridParaphraser()
    
    # Test text
    text = "Implementasi sistem informasi manajemen dapat meningkatkan efisiensi operasional perusahaan secara signifikan."
    
    print(f"📝 Text to Analyze:")
    print(f"   '{text}'")
    print()
    
    # Detailed analysis
    print("🔍 Performing detailed analysis...")
    result = paraphraser.paraphrase_with_analysis(text)
    
    print("=" * 50)
    print("DETAILED ANALYSIS RESULTS")
    print("=" * 50)
    
    print(f"📝 Original: '{result.original_text}'")
    print(f"📝 Paraphrased: '{result.paraphrased_text}'")
    print()
    
    print("📊 QUALITY METRICS:")
    print(f"   🎯 Overall Quality: {result.quality_score:.1f}/100")
    print(f"   🧠 Neural Confidence: {result.neural_confidence:.3f}")
    print(f"   🔗 Semantic Similarity: {result.semantic_similarity:.3f}")
    print(f"   📈 Lexical Diversity: {result.lexical_diversity:.3f}")
    print(f"   🔧 Syntactic Complexity: {result.syntactic_complexity:.3f}")
    print(f"   ✨ Fluency Score: {result.fluency_score:.3f}")
    print(f"   🔥 Confidence Score: {result.confidence_score:.3f}")
    print()
    
    print("📈 CHANGE STATISTICS:")
    print(f"   🔄 Word Changes: {result.word_changes}")
    print(f"   🏗️ Syntax Changes: {result.syntax_changes}")
    print(f"   ⏱️ Processing Time: {result.processing_time:.2f} seconds")
    print()
    
    print("🔧 TRANSFORMATIONS APPLIED:")
    for i, transform in enumerate(result.transformations_applied, 1):
        print(f"   {i}. {transform}")
    print()
    
    print("🎯 RECOMMENDATION:")
    if result.quality_score >= 85:
        print("   ✅ Excellent quality paraphrase")
    elif result.quality_score >= 70:
        print("   ✅ Good quality paraphrase")
    elif result.quality_score >= 50:
        print("   ⚠️ Acceptable quality paraphrase")
    else:
        print("   ❌ Low quality paraphrase")

def model_info_example():
    """Display model information"""
    print("=" * 60)
    print("MODEL INFORMATION EXAMPLE")
    print("=" * 60)
    
    # Initialize paraphraser
    paraphraser = IndoT5HybridParaphraser()
    
    # Get model info
    info = paraphraser.get_model_info()
    
    print("🔍 MODEL INFORMATION:")
    print(f"   🤖 Model Name: {info['model_name']}")
    print(f"   💻 Device: {info['device']}")
    print(f"   🚀 GPU Enabled: {info['use_gpu']}")
    print(f"   🔄 Synonym Rate: {info['synonym_rate']}")
    print(f"   🎯 Min Confidence: {info['min_confidence']}")
    print(f"   📊 Quality Threshold: {info['quality_threshold']}")
    print(f"   🔧 Max Transformations: {info['max_transformations']}")
    print(f"   📚 Synonyms Loaded: {info['synonyms_loaded']}")
    print(f"   🛑 Stopwords Loaded: {info['stopwords_loaded']}")
    print()

def main():
    """Run all examples"""
    print("🎯 INDOT5 HYBRID PARAPHRASER EXAMPLES")
    print("=" * 60)
    
    try:
        # Run all examples
        basic_example()
        method_comparison_example()
        multiple_variations_example()
        batch_processing_example()
        detailed_analysis_example()
        model_info_example()
        
        print("✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("   Make sure you have installed all dependencies:")
        print("   pip install -r requirements-neural.txt")

if __name__ == "__main__":
    main() 
