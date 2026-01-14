"""Test different paraphrase approaches"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load model
print("Loading IndoT5 model...")
tokenizer = AutoTokenizer.from_pretrained('Wikidepia/IndoT5-base', use_fast=False, legacy=True)
model = AutoModelForSeq2SeqLM.from_pretrained('Wikidepia/IndoT5-base')

# Test text
test_texts = [
    'Saya sedang belajar bahasa Indonesia dengan antusias.',
    'Penelitian ini menggunakan metode kualitatif.',
    'Teknologi berkembang sangat pesat di era modern.'
]

print('Testing T5 span corruption approach (mask and fill)...')
print('='*70)

for test_text in test_texts:
    print(f'\nOriginal: {test_text}')
    print('-'*70)
    
    # Approach 1: Use higher temperature and sampling for variation
    inputs = tokenizer(test_text, return_tensors='pt', max_length=512, truncation=True)
    
    # Try different generation strategies
    strategies = [
        {'num_beams': 5, 'do_sample': False, 'temperature': 1.0, 'name': 'Beam Search'},
        {'num_beams': 1, 'do_sample': True, 'temperature': 1.2, 'top_p': 0.9, 'name': 'Nucleus Sampling'},
        {'num_beams': 1, 'do_sample': True, 'temperature': 1.5, 'top_k': 50, 'name': 'Top-K Sampling'},
    ]
    
    for strategy in strategies:
        name = strategy.pop('name')
        outputs = model.generate(
            **inputs,
            max_length=128,
            early_stopping=True,
            **strategy
        )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f'{name}: {result}')
        strategy['name'] = name  # restore for next iteration

print('\n' + '='*70)
print('CONCLUSION: IndoT5-base needs different approach for paraphrasing')
