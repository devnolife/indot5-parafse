"""Test different prefixes for IndoT5 model"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model
print("Loading IndoT5 model...")
tokenizer = AutoTokenizer.from_pretrained('Wikidepia/IndoT5-base', use_fast=False, legacy=True)
model = AutoModelForSeq2SeqLM.from_pretrained('Wikidepia/IndoT5-base')

# Test text
test_text = 'Saya sedang belajar bahasa Indonesia dengan antusias.'

# Different prefixes to try
prefixes = [
    'paraphrase: ',
    'parafrasa: ',
    'parafrase: ',
    'ubah kalimat: ',
    'tulis ulang: ',
    '',
]

print('Testing different prefixes for IndoT5...')
print('='*60)
print(f'Original: {test_text}')
print('='*60)

for prefix in prefixes:
    input_text = prefix + test_text
    inputs = tokenizer(input_text, return_tensors='pt', max_length=512, truncation=True)
    
    outputs = model.generate(
        **inputs,
        max_length=128,
        num_beams=4,
        do_sample=True,
        temperature=0.8,
        early_stopping=True
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f'Prefix: "{prefix}"')
    print(f'Output: {result}')
    print('-'*60)
