from flask import Flask, request, jsonify, send_file
import os
import sys
import json
import uuid
from typing import Dict, Any
import logging
from werkzeug.utils import secure_filename

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import IndoT5HybridConfig, UPLOAD_DIR, ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH
from engines.indot5_hybrid_engine import IndoT5HybridParaphraser
from utils.file_parser import FileParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure upload
app.config['UPLOAD_FOLDER'] = str(UPLOAD_DIR)
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Global instances
paraphraser = None
file_parser = FileParser()

def initialize_paraphraser():
    """Initialize the paraphraser with default configuration"""
    global paraphraser
    try:
        config = IndoT5HybridConfig()
        paraphraser = IndoT5HybridParaphraser(
            model_name=config.model_name,
            use_gpu=config.use_gpu,
            synonym_rate=config.synonym_replacement_rate,
            min_confidence=config.neural_confidence_threshold,
            quality_threshold=config.min_quality_threshold,
            max_transformations=config.max_transformations_per_sentence,
            enable_caching=True
        )
        logger.info("Paraphraser initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize paraphraser: {e}")
        raise

@app.route('/')
def index():
    """Serve the HTML interface"""
    return send_file('index.html')

@app.route('/paraphrase', methods=['POST'])
def paraphrase():
    """Handle paraphrasing requests"""
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Extract parameters with defaults
        method = data.get('method', 'hybrid')
        num_variations = data.get('num_variations', 3)
        min_quality = data.get('min_quality', 0.6)
        max_length = data.get('max_length', 200)
        temperature = data.get('temperature', 1.0)
        
        # Validate parameters
        if method not in ['hybrid', 'neural', 'rule-based']:
            return jsonify({'error': 'Invalid method'}), 400
        
        if num_variations < 1 or num_variations > 10:
            return jsonify({'error': 'Number of variations must be between 1 and 10'}), 400
        
        if min_quality < 0 or min_quality > 1:
            return jsonify({'error': 'Min quality must be between 0 and 1'}), 400
        
        if max_length < 10 or max_length > 1000:
            return jsonify({'error': 'Max length must be between 10 and 1000'}), 400
        
        if temperature < 0.1 or temperature > 2.0:
            return jsonify({'error': 'Temperature must be between 0.1 and 2.0'}), 400
        
        logger.info(f"Processing text: {text[:50]}... with method: {method}")
        
        # Generate paraphrases using generate_variations for unique results
        results = paraphraser.generate_variations(text, num_variations=num_variations, method=method)
        paraphrases = []
        for result in results:
            paraphrases.append({
                'text': result.paraphrased_text,
                'quality_score': float(result.quality_score),
                'semantic_similarity': float(result.semantic_similarity),
                'lexical_diversity': float(result.lexical_diversity),
                'fluency_score': float(result.fluency_score),
                'method': result.method_used,
                'processing_time': float(result.processing_time),
                'transformations': result.transformations_applied,
                'word_changes': result.word_changes,
                'syntax_changes': result.syntax_changes,
                'success': result.success
            })
        
        # Sort by quality score (descending)
        paraphrases.sort(key=lambda x: x['quality_score'], reverse=True)
        
        response_data = {
            'original_text': text,
            'method': method,
            'paraphrases': paraphrases,
            'total_variations': len(paraphrases)
        }
        
        logger.info(f"Generated {len(paraphrases)} paraphrases")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error processing paraphrase request: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'paraphraser_ready': paraphraser is not None
    })

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and extract text"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed. Use PDF or TXT'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        try:
            # Extract text from file
            result = file_parser.process_file(file_path, chunk=True)
            
            # Clean up uploaded file
            os.remove(file_path)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'text': result['cleaned_text'],
                'chunks': result['chunks'],
                'chunk_count': result['chunk_count'],
                'metadata': result['metadata'],
                'character_count': len(result['cleaned_text'])
            })
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e
            
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/paraphrase-chunks', methods=['POST'])
def paraphrase_chunks():
    """Handle paraphrasing of multiple chunks (for large documents)"""
    try:
        data = request.get_json()
        
        if not data or 'chunks' not in data:
            return jsonify({'error': 'No chunks provided'}), 400
        
        chunks = data['chunks']
        method = data.get('method', 'hybrid')
        
        if not chunks:
            return jsonify({'error': 'Empty chunks array'}), 400
        
        results = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                result = paraphraser.paraphrase(chunk.strip(), method=method)
                results.append({
                    'chunk_index': i,
                    'original': chunk,
                    'paraphrased': result.paraphrased_text,
                    'quality_score': float(result.quality_score),
                    'success': result.success
                })
        
        # Combine all paraphrased chunks
        combined_text = ' '.join([r['paraphrased'] for r in results])
        avg_quality = sum(r['quality_score'] for r in results) / len(results) if results else 0
        
        return jsonify({
            'success': True,
            'chunks_processed': len(results),
            'combined_text': combined_text,
            'average_quality': avg_quality,
            'chunk_results': results
        })
        
    except Exception as e:
        logger.error(f"Error paraphrasing chunks: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    try:
        # Initialize the paraphraser
        print("Initializing IndoT5 Hybrid Paraphraser...")
        initialize_paraphraser()
        
        # Start the Flask app
        print("Starting Flask server...")
        print("Open your browser and go to: http://localhost:5000")
        print("Press Ctrl+C to stop the server")
        
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,  # Set to True for development
            threaded=True
        )
        
    except Exception as e:
        print(f"Failed to start server: {e}")
        sys.exit(1) 
