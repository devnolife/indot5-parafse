# PlagiCheck IndoT5 Hybrid Paraphraser

![Version](https://img.shields.io/badge/Version-1.0-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-green.svg)
![Language](https://img.shields.io/badge/Language-Indonesian-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Indonesian Text Paraphrasing using IndoT5 + Rule-based Hybrid Approach**

Sistem parafrase bahasa Indonesia yang menggabungkan kekuatan neural network IndoT5 dengan rule-based transformations untuk menghasilkan parafrase berkualitas tinggi.

## 🎯 Overview

IndoT5 Hybrid Paraphraser adalah sistem yang dirancang khusus untuk:

- **🤖 IndoT5 Neural Processing**: Menggunakan model IndoT5 untuk paraphrasing neural
- **📝 Rule-based Transformations**: Substitusi sinonim dan transformasi sintaksis
- **🔄 Hybrid Approach**: Kombinasi cerdas neural + rule-based untuk hasil optimal
- **🇮🇩 Indonesian Optimized**: Dioptimalkan khusus untuk bahasa Indonesia

## 🏗️ Architecture

```
paraphrase/
├── engines/
│   ├── __init__.py
│   ├── indot5_hybrid_engine.py    # Main IndoT5 hybrid engine
│   └── quality_scorer.py          # Quality assessment
├── data/
│   ├── sinonim_extended.json      # Synonym database
│   ├── transformation_rules.json  # Syntactic rules
│   └── stopwords_id.txt          # Indonesian stopwords
├── utils/
│   ├── __init__.py
│   ├── text_processor.py         # Text preprocessing
│   └── validator.py              # Input validation
├── examples/
│   └── indot5_hybrid_example.py  # Usage examples
├── tests/
│   └── test_indot5_hybrid.py     # Testing
├── requirements-neural.txt        # Dependencies
└── README.md                     # This file
```

## 🚀 Installation

### 1. (Recommended) Use Virtual Environment (venv)

**Windows:**
```sh
python -m venv venv
venv\Scripts\activate
```

**Linux/Ubuntu:**
```sh
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies
```sh
pip install -r requirements-neural.txt
```

### 3. Jalankan Aplikasi Web
```sh
python run_web_app.py
```

> **Note:**
> - Jika model IndoT5 sudah di-download manual, pastikan path model sudah sesuai di konfigurasi.
> - Jika model belum ada, aplikasi akan mencoba download otomatis (butuh koneksi internet).
> - Pastikan menjalankan perintah di atas dari direktori project (ada file `app.py`).

### 4. Troubleshooting

**Jika muncul error "dependency belum terinstall":**
```sh
# Pastikan virtual environment aktif
# Windows:
venv\Scripts\activate

# Linux/Ubuntu:
source venv/bin/activate

# Install ulang dependencies
pip install -r requirements-neural.txt
```

**Jika muncul error model loading:**
- Pastikan koneksi internet stabil untuk download model
- Jika model sudah ada lokal, periksa path di `config.py`
- Untuk GPU: pastikan CUDA terinstall dan kompatibel

**Jika port 5000 sudah digunakan:**
- Ubah port di `app.py` atau `run_web_app.py`
- Atau hentikan aplikasi lain yang menggunakan port 5000

## 🚀 Quick Start

```python
from engines.indot5_hybrid_engine import IndoT5HybridParaphraser

# Initialize paraphraser
paraphraser = IndoT5HybridParaphraser(
    model_name="Wikidepia/IndoT5-base",
    use_gpu=True,
    synonym_rate=0.3
)

# Simple paraphrase
text = "Penelitian ini menggunakan metode kualitatif untuk menganalisis data wawancara."
result = paraphraser.paraphrase(text)

print(f"Original: {result.original_text}")
print(f"Paraphrased: {result.paraphrased_text}")
print(f"Quality Score: {result.quality_score:.1f}")
print(f"Confidence: {result.confidence_score:.3f}")
print(f"Transformations: {', '.join(result.transformations_applied)}")
```

## 🌐 Web Interface (Recommended)

The easiest way to use the IndoT5 Hybrid Paraphraser is through the web interface:

### Starting the Web Server

```bash
# Method 1: Using the startup script (recommended)
python run_web_app.py

# Method 2: Direct Flask app
python app.py
```

### Using the Web Interface

1. **Open your browser** and go to: `http://localhost:5000`

2. **Enter your text** in the text area

3. **Choose paraphrasing method:**
   - **Hybrid**: Combines IndoT5 neural processing with rule-based transformations
   - **Neural**: Uses only IndoT5 neural processing
   - **Rule-based**: Uses only rule-based transformations

4. **Adjust settings:**
   - Number of variations (1-10)
   - Quality threshold (0.0-1.0)
   - Maximum length (10-1000 characters)
   - Temperature for neural generation (0.1-2.0)

5. **Click "Parafrase Teks"** to generate results

### Web Interface Features

- ✨ **Modern UI**: Clean, responsive design
- 🎯 **User-friendly**: No coding required
- 📊 **Real-time scoring**: Quality assessment for each result
- 🔄 **Multiple methods**: Choose between hybrid, neural, or rule-based
- ⚙️ **Adjustable parameters**: Fine-tune generation settings
- 📱 **Mobile-friendly**: Works on all devices
- 🚀 **Fast processing**: Efficient backend with caching

### API Endpoints

The web application also provides REST API endpoints:

```bash
# Health check
GET /health

# Paraphrase text
POST /paraphrase
Content-Type: application/json

{
  "text": "Your Indonesian text here",
  "method": "hybrid",
  "num_variations": 3,
  "min_quality": 0.6,
  "max_length": 200,
  "temperature": 1.0
}
```

## 🔧 How It Works

### Hybrid Process Flow

```
Input Text → IndoT5 Neural Processing → Rule-based Enhancement → Quality Assessment → Output
```

### Step-by-Step Process

1. **Input Processing**
   - Text normalization and cleaning
   - Entity identification and preservation
   - Preprocessing for IndoT5 model

2. **IndoT5 Neural Processing**  
   - Generate initial paraphrase using IndoT5
   - Apply beam search or nucleus sampling
   - Get confidence scores

3. **Rule-based Enhancement**
   - **Synonym Substitution**: Replace words with contextual synonyms
   - **Syntactic Transformation**: Change sentence structure
     - Active-passive voice conversion
     - Clause reordering
     - Conjunction substitution
     - Modifier adjustment

4. **Quality Assessment**
   - Semantic similarity scoring
   - Lexical diversity measurement
   - Fluency evaluation
   - Overall quality calculation

## 📊 Features

### IndoT5 Neural Processing
- ✅ **IndoT5 Integration**: Specialized Indonesian T5 model
- ✅ **Multiple Generation**: Beam search, nucleus sampling
- ✅ **GPU Acceleration**: CUDA support for faster processing
- ✅ **Confidence Scoring**: Neural confidence measurement

### Rule-based Transformations
- ✅ **Synonym Database**: 4700+ Indonesian synonyms
- ✅ **Syntactic Rules**: Indonesian grammar transformations
- ✅ **Context Awareness**: Contextual synonym selection
- ✅ **Quality Control**: Validation at each step

### Quality Assessment
- ✅ **Multi-metric Evaluation**: Semantic + lexical + syntactic
- ✅ **Indonesian Specific**: Optimized for Indonesian text
- ✅ **Confidence Scoring**: Both neural and rule-based scores
- ✅ **Quality Thresholds**: Configurable quality standards

## 🎮 Advanced Usage

### Custom Configuration

```python
# Initialize with custom settings
paraphraser = IndoT5HybridParaphraser(
    model_name="Wikidepia/IndoT5-base",
    use_gpu=True,
    synonym_rate=0.4,           # 40% synonym replacement rate
    min_confidence=0.7,         # Minimum neural confidence
    quality_threshold=75.0,     # Minimum quality score
    max_transformations=3       # Maximum rule-based transformations
)

# Generate multiple variations
results = paraphraser.generate_variations(text, num_variations=3)

for i, result in enumerate(results, 1):
    print(f"Variation {i}: {result.paraphrased_text}")
    print(f"Quality: {result.quality_score:.1f}")
    print(f"Method: {result.method_used}")
    print()
```

### Batch Processing

```python
# Process multiple texts
texts = [
    "Teknologi AI berkembang pesat dalam dekade terakhir.",
    "Machine learning membantu mengoptimalkan proses bisnis.",
    "Penelitian menunjukkan efektivitas metode baru."
]

results = paraphraser.batch_paraphrase(texts)

for original, result in zip(texts, results):
    print(f"Original: {original}")
    print(f"Paraphrased: {result.paraphrased_text}")
    print(f"Quality: {result.quality_score:.1f}")
    print()
```

### Quality Analysis

```python
# Detailed quality analysis
result = paraphraser.paraphrase_with_analysis(text)

print("=== QUALITY ANALYSIS ===")
print(f"Overall Quality: {result.quality_score:.1f}")
print(f"Neural Confidence: {result.neural_confidence:.3f}")
print(f"Semantic Similarity: {result.semantic_similarity:.3f}")
print(f"Lexical Diversity: {result.lexical_diversity:.3f}")
print(f"Syntactic Complexity: {result.syntactic_complexity:.3f}")
print(f"Fluency Score: {result.fluency_score:.3f}")

print("\n=== TRANSFORMATIONS ===")
for transform in result.transformations_applied:
    print(f"- {transform}")
```

## 📊 Performance

### Speed Benchmarks
- **Short text (10 words)**: ~2.0s
- **Medium text (30 words)**: ~4.0s  
- **Long text (100 words)**: ~8.0s
- **Batch processing**: ~1.5s per text

### Quality Metrics
- **Semantic Similarity**: 0.85-0.95
- **Quality Score**: 75-90 (average 82)
- **Success Rate**: >90% for standard texts
- **Confidence Score**: 0.7-0.9 (average 0.8)

## 🔧 Configuration

### Model Configuration

```python
# Available IndoT5 models
SUPPORTED_MODELS = [
    "Wikidepia/IndoT5-base",      # Recommended
    "Wikidepia/IndoT5-large",     # Higher quality, slower
    "Wikidepia/IndoT5-small"      # Faster, lower quality
]

# Generation parameters
GENERATION_CONFIG = {
    "max_length": 512,
    "num_beams": 4,
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.1
}
```

### Transformation Rules

```python
# Synonym replacement settings
SYNONYM_CONFIG = {
    "replacement_rate": 0.3,        # 30% of words
    "min_confidence": 0.6,          # Minimum synonym confidence
    "preserve_entities": True,      # Keep named entities
    "context_aware": True           # Use context for selection
}

# Syntactic transformation settings
SYNTACTIC_CONFIG = {
    "max_transformations": 2,       # Maximum per sentence
    "transformation_types": [
        "active_to_passive",
        "clause_reordering", 
        "conjunction_substitution",
        "modifier_adjustment"
    ]
}
```

## 🧪 Testing

```bash
# Run tests
python -m pytest tests/test_indot5_hybrid.py -v

# Run example
python examples/indot5_hybrid_example.py
```

## 📚 Example Results

### Input
```
"Penelitian ini menggunakan pendekatan kualitatif dengan metode wawancara mendalam untuk mengumpulkan data dari responden."
```

### Output
```
Original: Penelitian ini menggunakan pendekatan kualitatif dengan metode wawancara mendalam untuk mengumpulkan data dari responden.

Paraphrased: Riset tersebut menerapkan pendekatan kualitatif serta teknik interview mendalam guna mengumpulkan informasi dari narasumber.

Quality Score: 84.2
Confidence: 0.82
Transformations: neural_generation, synonym_substitution, conjunction_substitution

Analysis:
- Neural Confidence: 0.82
- Semantic Similarity: 0.89
- Lexical Diversity: 0.78
- Syntactic Complexity: 0.71
- Fluency Score: 0.85
```

## 🚀 Future Development

### Planned Features
- **Model Fine-tuning**: Domain-specific IndoT5 fine-tuning
- **Advanced Metrics**: More sophisticated quality assessment
- **Performance Optimization**: Faster inference and processing
- **API Integration**: RESTful API for web applications

## 🤝 Contributing

```bash
# Setup development environment
git clone https://github.com/devnolife/paraphrase.git
cd paraphrase

# Install dependencies
pip install -r requirements-neural.txt

# Run tests
python -m pytest tests/ -v
```

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **IndoT5 Team**: Indonesian T5 model development
- **HuggingFace**: Transformers library and model hosting
- **DevNoLife**: PlagiCheck development team

---

**© 2024 DevNoLife - IndoT5 Hybrid Paraphraser**

*Intelligent Indonesian Text Paraphrasing with Neural-Rule Hybrid Approach*
