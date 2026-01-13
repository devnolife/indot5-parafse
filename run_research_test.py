"""
Skrip Pengujian Penelitian - IndoT5 Hybrid Paraphraser
Menjalankan pengujian komprehensif dan menyimpan hasil ke folder 'hasil/'
untuk dokumentasi penelitian
"""

import sys
import os
import json
import time
import csv
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import asdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engines.indot5_hybrid_engine import IndoT5HybridParaphraser, IndoT5HybridResult
from config import config, IndoT5HybridConfig

# ============================================================================
# KONFIGURASI PENELITIAN
# ============================================================================

HASIL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hasil")
EXPERIMENTS_DIR = os.path.join(HASIL_DIR, "experiments")
BATCH_DIR = os.path.join(HASIL_DIR, "batch_results")
COMPARISON_DIR = os.path.join(HASIL_DIR, "comparison_reports")
QUALITY_DIR = os.path.join(HASIL_DIR, "quality_analysis")

# Dataset pengujian - Kalimat bahasa Indonesia untuk parafrase
TEST_SENTENCES = [
    "Penelitian ini menggunakan metode kualitatif untuk menganalisis data wawancara dari responden.",
    "Teknologi artificial intelligence berkembang pesat dalam dekade terakhir dan memberikan dampak signifikan.",
    "Pemerintah Indonesia mencanangkan program digitalisasi untuk meningkatkan pelayanan publik.",
    "Mahasiswa diharapkan mampu mengembangkan kemampuan berpikir kritis melalui kegiatan pembelajaran.",
    "Perubahan iklim global mempengaruhi pola cuaca dan mengancam ketahanan pangan di berbagai negara.",
    "Penggunaan media sosial telah mengubah cara masyarakat berkomunikasi dan berinteraksi.",
    "Hasil penelitian menunjukkan bahwa motivasi belajar berpengaruh terhadap prestasi akademik siswa.",
    "Ekonomi digital membuka peluang baru bagi pelaku usaha mikro kecil dan menengah.",
    "Pendidikan karakter menjadi fokus utama dalam kurikulum pendidikan nasional saat ini.",
    "Inovasi teknologi informasi membantu meningkatkan efisiensi dalam berbagai sektor industri."
]

# ============================================================================
# FUNGSI UTILITAS
# ============================================================================

def ensure_directories():
    """Pastikan semua direktori hasil tersedia"""
    for dir_path in [HASIL_DIR, EXPERIMENTS_DIR, BATCH_DIR, COMPARISON_DIR, QUALITY_DIR]:
        os.makedirs(dir_path, exist_ok=True)

def get_timestamp():
    """Mendapatkan timestamp untuk penamaan file"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def result_to_dict(result: IndoT5HybridResult) -> Dict[str, Any]:
    """Konversi hasil parafrase ke dictionary"""
    return {
        "original_text": result.original_text,
        "paraphrased_text": result.paraphrased_text,
        "method_used": result.method_used,
        "transformations_applied": result.transformations_applied,
        "quality_score": float(round(result.quality_score, 2)),
        "confidence_score": float(round(result.confidence_score, 4)),
        "neural_confidence": float(round(result.neural_confidence, 4)),
        "semantic_similarity": float(round(result.semantic_similarity, 4)),
        "lexical_diversity": float(round(result.lexical_diversity, 4)),
        "syntactic_complexity": float(round(result.syntactic_complexity, 4)),
        "fluency_score": float(round(result.fluency_score, 4)),
        "processing_time": float(round(result.processing_time, 4)),
        "word_changes": int(result.word_changes),
        "syntax_changes": int(result.syntax_changes),
        "success": result.success,
        "error_message": result.error_message,
        "alternatives": result.alternatives
    }

def print_header(title: str):
    """Print header yang terformat"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def print_result_summary(result: IndoT5HybridResult, index: int = None):
    """Print ringkasan hasil parafrase"""
    prefix = f"[{index}] " if index else ""
    print(f"\n{prefix}📝 Original: {result.original_text[:60]}...")
    print(f"   ✅ Paraphrase: {result.paraphrased_text[:60]}...")
    print(f"   📊 Quality: {result.quality_score:.1f} | Semantic: {result.semantic_similarity:.3f} | Time: {result.processing_time:.2f}s")

# ============================================================================
# FUNGSI PENGUJIAN
# ============================================================================

def run_single_experiment(paraphraser: IndoT5HybridParaphraser, 
                          text: str, 
                          method: str = "hybrid") -> Dict[str, Any]:
    """
    Menjalankan eksperimen tunggal
    """
    experiment_id = f"EXP-{get_timestamp()}"
    
    start_time = time.time()
    result = paraphraser.paraphrase(text, method=method)
    total_time = time.time() - start_time
    
    experiment_data = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "model_name": paraphraser.model_name,
            "method": method,
            "synonym_rate": paraphraser.synonym_rate,
            "quality_threshold": paraphraser.quality_threshold,
            "use_gpu": paraphraser.use_gpu
        },
        "input": {
            "text": text,
            "word_count": len(text.split())
        },
        "result": result_to_dict(result),
        "total_processing_time": round(total_time, 4)
    }
    
    return experiment_data

def run_batch_experiments(paraphraser: IndoT5HybridParaphraser,
                          sentences: List[str],
                          method: str = "hybrid") -> Dict[str, Any]:
    """
    Menjalankan batch eksperimen untuk multiple kalimat
    """
    batch_id = f"BATCH-{get_timestamp()}"
    
    print_header(f"BATCH EXPERIMENT: {method.upper()}")
    print(f"📊 Total kalimat: {len(sentences)}")
    print(f"🔧 Metode: {method}")
    
    results = []
    total_quality = 0
    total_semantic = 0
    total_time = 0
    success_count = 0
    
    for i, sentence in enumerate(sentences, 1):
        print(f"\n🔄 Processing [{i}/{len(sentences)}]...")
        
        experiment = run_single_experiment(paraphraser, sentence, method)
        results.append(experiment)
        
        if experiment["result"]["success"]:
            success_count += 1
            total_quality += experiment["result"]["quality_score"]
            total_semantic += experiment["result"]["semantic_similarity"]
        
        total_time += experiment["total_processing_time"]
        
        print_result_summary(
            IndoT5HybridResult(
                original_text=sentence,
                paraphrased_text=experiment["result"]["paraphrased_text"],
                method_used=method,
                transformations_applied=experiment["result"]["transformations_applied"],
                quality_score=experiment["result"]["quality_score"],
                confidence_score=experiment["result"]["confidence_score"],
                neural_confidence=experiment["result"]["neural_confidence"],
                semantic_similarity=experiment["result"]["semantic_similarity"],
                lexical_diversity=experiment["result"]["lexical_diversity"],
                syntactic_complexity=experiment["result"]["syntactic_complexity"],
                fluency_score=experiment["result"]["fluency_score"],
                processing_time=experiment["result"]["processing_time"],
                word_changes=experiment["result"]["word_changes"],
                syntax_changes=experiment["result"]["syntax_changes"],
                success=experiment["result"]["success"]
            ),
            index=i
        )
    
    # Hitung statistik
    avg_quality = total_quality / success_count if success_count > 0 else 0
    avg_semantic = total_semantic / success_count if success_count > 0 else 0
    
    batch_data = {
        "batch_id": batch_id,
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "model_name": paraphraser.model_name,
            "method": method,
            "synonym_rate": float(paraphraser.synonym_rate),
            "quality_threshold": float(paraphraser.quality_threshold)
        },
        "summary": {
            "total_sentences": len(sentences),
            "success_count": success_count,
            "success_rate": float(round(success_count / len(sentences) * 100, 2)),
            "average_quality_score": float(round(avg_quality, 2)),
            "average_semantic_similarity": float(round(avg_semantic, 4)),
            "total_processing_time": float(round(total_time, 2)),
            "average_processing_time": float(round(total_time / len(sentences), 2))
        },
        "results": results
    }
    
    print(f"\n📊 BATCH SUMMARY:")
    print(f"   ✅ Success Rate: {batch_data['summary']['success_rate']}%")
    print(f"   📈 Avg Quality: {batch_data['summary']['average_quality_score']}")
    print(f"   🔗 Avg Semantic: {batch_data['summary']['average_semantic_similarity']}")
    print(f"   ⏱️ Total Time: {batch_data['summary']['total_processing_time']}s")
    
    return batch_data

def run_method_comparison(paraphraser: IndoT5HybridParaphraser,
                          sentences: List[str]) -> Dict[str, Any]:
    """
    Menjalankan perbandingan antara metode hybrid, neural, dan rule-based
    """
    comparison_id = f"COMPARE-{get_timestamp()}"
    
    print_header("METHOD COMPARISON EXPERIMENT")
    print(f"📊 Comparing: hybrid vs neural vs rule-based")
    print(f"📝 Test sentences: {len(sentences)}")
    
    methods = ["hybrid", "neural", "rule-based"]
    comparison_results = {}
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"🔧 Testing Method: {method.upper()}")
        print('='*50)
        
        batch_result = run_batch_experiments(paraphraser, sentences, method)
        comparison_results[method] = {
            "summary": batch_result["summary"],
            "sample_results": [r["result"] for r in batch_result["results"][:3]]  # Top 3 samples
        }
    
    # Generate comparison summary
    comparison_data = {
        "comparison_id": comparison_id,
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "model_name": paraphraser.model_name,
            "synonym_rate": float(paraphraser.synonym_rate),
            "quality_threshold": float(paraphraser.quality_threshold)
        },
        "test_sentences_count": len(sentences),
        "method_comparison": comparison_results,
        "analysis": {
            "best_quality": max(methods, key=lambda m: comparison_results[m]["summary"]["average_quality_score"]),
            "best_semantic": max(methods, key=lambda m: comparison_results[m]["summary"]["average_semantic_similarity"]),
            "fastest": min(methods, key=lambda m: comparison_results[m]["summary"]["average_processing_time"]),
            "highest_success": max(methods, key=lambda m: comparison_results[m]["summary"]["success_rate"])
        }
    }
    
    print_header("COMPARISON ANALYSIS")
    print(f"🏆 Best Quality: {comparison_data['analysis']['best_quality']}")
    print(f"🔗 Best Semantic: {comparison_data['analysis']['best_semantic']}")
    print(f"⚡ Fastest: {comparison_data['analysis']['fastest']}")
    print(f"✅ Highest Success: {comparison_data['analysis']['highest_success']}")
    
    return comparison_data

def generate_quality_report(batch_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate laporan kualitas dari hasil batch
    """
    results = batch_results.get("results", [])
    
    quality_scores = []
    semantic_scores = []
    lexical_scores = []
    fluency_scores = []
    processing_times = []
    
    for exp in results:
        r = exp["result"]
        if r["success"]:
            quality_scores.append(r["quality_score"])
            semantic_scores.append(r["semantic_similarity"])
            lexical_scores.append(r["lexical_diversity"])
            fluency_scores.append(r["fluency_score"])
            processing_times.append(r["processing_time"])
    
    def calc_stats(scores):
        if not scores:
            return {"min": 0, "max": 0, "avg": 0, "std": 0}
        import statistics
        return {
            "min": float(round(min(scores), 4)),
            "max": float(round(max(scores), 4)),
            "avg": float(round(statistics.mean(scores), 4)),
            "std": float(round(statistics.stdev(scores), 4)) if len(scores) > 1 else 0
        }
    
    quality_report = {
        "report_id": f"QUALITY-{get_timestamp()}",
        "timestamp": datetime.now().isoformat(),
        "source_batch": batch_results.get("batch_id", "unknown"),
        "total_samples": len(results),
        "successful_samples": len(quality_scores),
        "metrics": {
            "quality_score": calc_stats(quality_scores),
            "semantic_similarity": calc_stats(semantic_scores),
            "lexical_diversity": calc_stats(lexical_scores),
            "fluency_score": calc_stats(fluency_scores),
            "processing_time": calc_stats(processing_times)
        }
    }
    
    return quality_report

def save_to_csv(batch_results: Dict[str, Any], filename: str):
    """
    Simpan hasil ke format CSV untuk analisis di Excel/SPSS
    """
    results = batch_results.get("results", [])
    
    csv_path = os.path.join(QUALITY_DIR, filename)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            "no", "original_text", "paraphrased_text", "method",
            "quality_score", "semantic_similarity", "lexical_diversity",
            "fluency_score", "processing_time", "word_changes", "syntax_changes", "success"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, exp in enumerate(results, 1):
            r = exp["result"]
            writer.writerow({
                "no": i,
                "original_text": exp["input"]["text"],
                "paraphrased_text": r["paraphrased_text"],
                "method": r["method_used"],
                "quality_score": r["quality_score"],
                "semantic_similarity": r["semantic_similarity"],
                "lexical_diversity": r["lexical_diversity"],
                "fluency_score": r["fluency_score"],
                "processing_time": r["processing_time"],
                "word_changes": r["word_changes"],
                "syntax_changes": r["syntax_changes"],
                "success": r["success"]
            })
    
    print(f"✅ CSV saved: {csv_path}")
    return csv_path

# ============================================================================
# FUNGSI UTAMA
# ============================================================================

def main():
    """
    Fungsi utama untuk menjalankan pengujian penelitian
    """
    print_header("PENGUJIAN PENELITIAN - INDOT5 HYBRID PARAPHRASER")
    print(f"📅 Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Hasil disimpan di: {HASIL_DIR}")
    
    # Pastikan direktori tersedia
    ensure_directories()
    
    # Inisialisasi paraphraser
    print("\n🚀 Initializing IndoT5 Hybrid Paraphraser...")
    try:
        paraphraser = IndoT5HybridParaphraser(
            model_name="Wikidepia/IndoT5-base",
            use_gpu=True,
            synonym_rate=0.3,
            min_confidence=0.7,
            quality_threshold=60.0
        )
        print("✅ Paraphraser initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing paraphraser: {e}")
        print("⚠️ Pastikan semua dependencies terinstall: pip install -r requirements-neural.txt")
        return
    
    timestamp = get_timestamp()
    
    # =========================================================================
    # 1. EKSPERIMEN TUNGGAL
    # =========================================================================
    print_header("1. EKSPERIMEN TUNGGAL")
    single_experiment = run_single_experiment(
        paraphraser, 
        TEST_SENTENCES[0], 
        method="hybrid"
    )
    
    # Simpan hasil eksperimen tunggal
    single_file = os.path.join(EXPERIMENTS_DIR, f"experiment_{timestamp}.json")
    with open(single_file, 'w', encoding='utf-8') as f:
        json.dump(single_experiment, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved: {single_file}")
    
    # =========================================================================
    # 2. BATCH EXPERIMENT - HYBRID METHOD
    # =========================================================================
    print_header("2. BATCH EXPERIMENT - HYBRID METHOD")
    batch_hybrid = run_batch_experiments(paraphraser, TEST_SENTENCES, method="hybrid")
    
    # Simpan hasil batch
    batch_file = os.path.join(BATCH_DIR, f"batch_hybrid_{timestamp}.json")
    with open(batch_file, 'w', encoding='utf-8') as f:
        json.dump(batch_hybrid, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved: {batch_file}")
    
    # Simpan ke CSV
    save_to_csv(batch_hybrid, f"results_hybrid_{timestamp}.csv")
    
    # =========================================================================
    # 3. QUALITY ANALYSIS
    # =========================================================================
    print_header("3. QUALITY ANALYSIS")
    quality_report = generate_quality_report(batch_hybrid)
    
    print(f"\n📊 QUALITY METRICS SUMMARY:")
    for metric, stats in quality_report["metrics"].items():
        print(f"   {metric}: min={stats['min']}, max={stats['max']}, avg={stats['avg']}, std={stats['std']}")
    
    # Simpan laporan kualitas
    quality_file = os.path.join(QUALITY_DIR, f"quality_report_{timestamp}.json")
    with open(quality_file, 'w', encoding='utf-8') as f:
        json.dump(quality_report, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved: {quality_file}")
    
    # =========================================================================
    # 4. METHOD COMPARISON (Opsional - uncomment untuk menjalankan)
    # =========================================================================
    print_header("4. METHOD COMPARISON")
    print("⚠️ Perbandingan metode menggunakan 3 kalimat pertama untuk efisiensi waktu")
    
    comparison = run_method_comparison(paraphraser, TEST_SENTENCES[:3])
    
    # Simpan hasil perbandingan
    comparison_file = os.path.join(COMPARISON_DIR, f"comparison_{timestamp}.json")
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved: {comparison_file}")
    
    # =========================================================================
    # RINGKASAN AKHIR
    # =========================================================================
    print_header("RINGKASAN PENGUJIAN")
    print(f"""
📁 File hasil penelitian tersimpan di:

   📂 {EXPERIMENTS_DIR}/
      └── experiment_{timestamp}.json
   
   📂 {BATCH_DIR}/
      └── batch_hybrid_{timestamp}.json
   
   📂 {QUALITY_DIR}/
      ├── quality_report_{timestamp}.json
      └── results_hybrid_{timestamp}.csv
   
   📂 {COMPARISON_DIR}/
      └── comparison_{timestamp}.json

📊 Statistik Pengujian:
   • Total kalimat diuji: {len(TEST_SENTENCES)}
   • Success rate: {batch_hybrid['summary']['success_rate']}%
   • Average quality score: {batch_hybrid['summary']['average_quality_score']}
   • Average semantic similarity: {batch_hybrid['summary']['average_semantic_similarity']}
   • Total processing time: {batch_hybrid['summary']['total_processing_time']}s

✅ Pengujian selesai! Hasil siap untuk dokumentasi penelitian.
    """)

if __name__ == "__main__":
    main()
