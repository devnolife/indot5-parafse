"""Test IndoT5 paraphrase dengan perbaikan baru"""
import sys
sys.path.append('.')

from engines.indot5_hybrid_engine import IndoT5HybridParaphraser

print("Loading IndoT5 Hybrid Paraphraser...")
paraphraser = IndoT5HybridParaphraser(
    model_name="Wikidepia/IndoT5-base",
    use_gpu=False,
    synonym_rate=0.4
)

test_texts = [
    "Saya sedang belajar bahasa Indonesia dengan antusias.",
    "Penelitian ini menggunakan metode kualitatif untuk menganalisis data.",
    "Teknologi berkembang sangat pesat di era modern ini.",
    "Pendidikan merupakan kunci utama untuk membangun masa depan yang cerah."
]

print("\n" + "="*70)
print("TESTING INDOT5 HYBRID PARAPHRASER")
print("="*70)

for text in test_texts:
    print(f"\n📝 Original: {text}")
    print("-"*70)
    
    # Test all methods
    methods = ['hybrid', 'neural', 'rule-based']
    
    for method in methods:
        result = paraphraser.paraphrase(text, method=method)
        print(f"\n🔄 Method: {method.upper()}")
        print(f"   Result: {result.paraphrased_text}")
        print(f"   Quality: {result.quality_score:.1f}%")
        print(f"   Transformations: {', '.join(result.transformations_applied[:3])}")
    
    print("="*70)

print("\n✅ Testing complete!")
