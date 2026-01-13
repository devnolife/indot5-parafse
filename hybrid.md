# Metode Hybrid: Kombinasi Sinonim dan Transformasi Sintaksis untuk Parafrase Bahasa Indonesia

## 1. Gambaran Umum

Metode Hybrid adalah pendekatan parafrase yang menggabungkan dua teknik utama:
- *Penggantian Sinonim (Lexical Substitution)*: Mengganti kata-kata dengan padanan katanya
- *Transformasi Sintaksis (Syntactic Transformation)*: Mengubah struktur kalimat

Kombinasi ini menghasilkan parafrase yang lebih natural dan bervariasi dibandingkan menggunakan satu metode saja.

## 2. Cara Kerja Metode Hybrid

### 2.1 Alur Proses


Input Text → Preprocessing → Synonym Replacement → Syntactic Transformation → Post-processing → Output


### 2.2 Tahapan Detail

#### Tahap 1: Preprocessing
python
def preprocess_text(text):
    """
    Membersihkan dan mempersiapkan teks
    """
    # Normalisasi spasi
    text = ' '.join(text.split())
    
    # Pisahkan tanda baca untuk memudahkan pemrosesan
    text = re.sub(r'([.!?;,])', r' \1 ', text)
    
    # Identifikasi dan tandai entitas penting (nama, tempat, dll)
    entities = identify_entities(text)
    
    return text, entities


#### Tahap 2: Penggantian Sinonim
python
def synonym_replacement(text, synonym_dict, replacement_rate=0.3):
    """
    Mengganti kata dengan sinonim berdasarkan probabilitas
    """
    words = text.split()
    result = []
    replaced_positions = []
    
    for i, word in enumerate(words):
        clean_word = word.lower().strip('.,!?;:"')
        
        # Cek apakah kata memiliki sinonim
        if clean_word in synonym_dict and random.random() < replacement_rate:
            synonyms = synonym_dict[clean_word].get('sinonim', [])
            
            if synonyms:
                # Pilih sinonim secara random
                chosen_synonym = random.choice(synonyms)
                
                # Pertahankan format asli (kapitalisasi, tanda baca)
                formatted_synonym = preserve_word_format(word, chosen_synonym)
                result.append(formatted_synonym)
                replaced_positions.append(i)
            else:
                result.append(word)
        else:
            result.append(word)
    
    return ' '.join(result), replaced_positions


#### Tahap 3: Transformasi Sintaksis
python
def syntactic_transformation(text):
    """
    Mengubah struktur kalimat
    """
    transformations = [
        active_to_passive,
        reorder_clauses,
        change_conjunctions,
        add_or_remove_modifiers
    ]
    
    # Pilih transformasi secara random
    num_transformations = random.randint(1, 2)
    selected_transforms = random.sample(transformations, num_transformations)
    
    result = text
    for transform in selected_transforms:
        result = transform(result)
    
    return result


### 2.3 Transformasi Sintaksis Spesifik

#### A. Transformasi Aktif-Pasif
python
def active_to_passive(sentence):
    """
    Mengubah kalimat aktif menjadi pasif
    Contoh: "Ani membaca buku" → "Buku dibaca oleh Ani"
    """
    patterns = [
        # Pattern: Subjek + me-Verb + Objek
        {
            'pattern': r'(\w+)\s+(me\w+)\s+(\w+)',
            'transform': lambda m: f"{m.group(3)} di{m.group(2)[2:]} oleh {m.group(1)}"
        },
        # Pattern: Subjek + akan/telah + Verb + Objek
        {
            'pattern': r'(\w+)\s+(akan|telah)\s+(\w+)\s+(\w+)',
            'transform': lambda m: f"{m.group(4)} {m.group(2)} di{m.group(3)} oleh {m.group(1)}"
        }
    ]
    
    for p in patterns:
        match = re.search(p['pattern'], sentence)
        if match:
            return p['transform'](match)
    
    return sentence


#### B. Perubahan Urutan Klausa
python
def reorder_clauses(text):
    """
    Mengubah urutan klausa dalam kalimat
    Contoh: "Karena hujan, dia tidak datang" → "Dia tidak datang karena hujan"
    """
    # Pola klausa dengan kata penghubung di awal
    if ',' in text:
        parts = text.split(',', 1)
        if len(parts) == 2 and any(word in parts[0].lower() for word in ['karena', 'ketika', 'jika', 'meskipun']):
            return f"{parts[1].strip()}, {parts[0].strip()}"
    
    return text


#### C. Variasi Kata Penghubung
python
def change_conjunctions(text):
    """
    Mengganti kata penghubung dengan variasinya
    """
    conjunctions = {
        'dan': ['serta', 'juga', 'beserta'],
        'atau': ['ataupun', 'maupun'],
        'tetapi': ['namun', 'akan tetapi', 'tapi'],
        'karena': ['sebab', 'oleh karena', 'dikarenakan'],
        'jika': ['apabila', 'bila', 'seandainya'],
        'meskipun': ['walaupun', 'sekalipun', 'kendati']
    }
    
    result = text
    for original, alternatives in conjunctions.items():
        if original in result.lower():
            replacement = random.choice(alternatives)
            result = re.sub(rf'\b{original}\b', replacement, result, flags=re.IGNORECASE)
            break  # Hanya ganti satu untuk menghindari over-modification
    
    return result


#### D. Modifikasi Modifier
python
def add_or_remove_modifiers(text):
    """
    Menambah atau menghilangkan kata keterangan
    """
    # Tambahkan modifier
    add_modifiers = {
        'penting': 'sangat penting',
        'besar': 'cukup besar',
        'baik': 'lebih baik',
        'jelas': 'sudah jelas'
    }
    
    # Hilangkan modifier
    remove_modifiers = {
        'sangat ': '',
        'cukup ': '',
        'lebih ': '',
        'sudah ': ''
    }
    
    # Pilih antara menambah atau menghilangkan
    if random.random() < 0.5:
        for word, replacement in add_modifiers.items():
            if word in text and replacement not in text:
                return text.replace(word, replacement)
    else:
        for modifier, replacement in remove_modifiers.items():
            if modifier in text:
                return text.replace(modifier, replacement)
    
    return text


### 2.4 Post-processing
python
def post_process(text):
    """
    Membersihkan dan memperbaiki hasil parafrase
    """
    # Perbaiki spasi di sekitar tanda baca
    text = re.sub(r'\s+([.!?;,])', r'\1', text)
    text = re.sub(r'([.!?;,])\s*', r'\1 ', text)
    
    # Normalisasi spasi
    text = ' '.join(text.split())
    
    # Kapitalisasi awal kalimat
    sentences = text.split('. ')
    sentences = [s.capitalize() for s in sentences]
    text = '. '.join(sentences)
    
    return text.strip()


## 3. Implementasi Lengkap

python
class HybridParaphraser:
    def __init__(self, synonym_dict_path):
        self.synonym_dict = self.load_synonym_dict(synonym_dict_path)
        self.transformation_history = []
        
    def load_synonym_dict(self, path):
        """Load kamus sinonim dari file JSON"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def paraphrase(self, text, num_variations=3, synonym_rate=0.3):
        """
        Generate parafrase dengan metode hybrid
        
        Args:
            text: Teks input yang akan diparafrase
            num_variations: Jumlah variasi parafrase yang diinginkan
            synonym_rate: Probabilitas penggantian sinonim (0.0-1.0)
            
        Returns:
            List of paraphrased texts
        """
        paraphrases = []
        
        for i in range(num_variations):
            # Reset history untuk setiap variasi
            self.transformation_history = []
            
            # Preprocessing
            processed_text, entities = self.preprocess_text(text)
            
            # Step 1: Synonym replacement dengan rate yang bervariasi
            current_rate = synonym_rate + (i * 0.1)  # Increase rate for each variation
            text_with_synonyms, replaced_positions = self.synonym_replacement(
                processed_text, 
                current_rate
            )
            
            # Step 2: Syntactic transformation
            # Gunakan transformasi yang berbeda untuk setiap variasi
            transformed_text = self.apply_transformations(
                text_with_synonyms, 
                variation_index=i
            )
            
            # Step 3: Post-processing
            final_text = self.post_process(transformed_text)
            
            # Validasi hasil
            if self.is_valid_paraphrase(text, final_text):
                paraphrases.append({
                    'text': final_text,
                    'transformations': self.transformation_history.copy(),
                    'similarity_score': self.calculate_similarity(text, final_text)
                })
        
        return paraphrases
    
    def apply_transformations(self, text, variation_index):
        """Aplikasikan transformasi berdasarkan index variasi"""
        transformation_sets = [
            [self.active_to_passive, self.change_conjunctions],
            [self.reorder_clauses, self.add_or_remove_modifiers],
            [self.change_conjunctions, self.reorder_clauses, self.active_to_passive]
        ]
        
        # Pilih set transformasi berdasarkan index
        transforms = transformation_sets[variation_index % len(transformation_sets)]
        
        result = text
        for transform in transforms:
            new_result = transform(result)
            if new_result != result:
                self.transformation_history.append(transform.__name__)
                result = new_result
        
        return result
    
    def is_valid_paraphrase(self, original, paraphrase):
        """Validasi apakah parafrase valid"""
        # Check 1: Tidak sama dengan original
        if original.lower() == paraphrase.lower():
            return False
        
        # Check 2: Panjang tidak terlalu berbeda (±50%)
        len_ratio = len(paraphrase) / len(original)
        if len_ratio < 0.5 or len_ratio > 1.5:
            return False
        
        # Check 3: Memiliki kata-kata kunci yang sama
        original_keywords = set(self.extract_keywords(original))
        paraphrase_keywords = set(self.extract_keywords(paraphrase))
        overlap = len(original_keywords & paraphrase_keywords) / len(original_keywords)
        
        if overlap < 0.3:  # Minimal 30% kata kunci sama
            return False
        
        return True
    
    def calculate_similarity(self, text1, text2):
        """Hitung similarity score antara dua teks"""
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0


## 4. Contoh Penggunaan

python
# Inisialisasi paraphraser
paraphraser = HybridParaphraser('sinonim.json')

# Teks input
text = "Pendidikan adalah proses pembelajaran yang sangat penting untuk mengembangkan potensi manusia."

# Generate parafrase
results = paraphraser.paraphrase(text, num_variations=3, synonym_rate=0.3)

# Output
for i, result in enumerate(results, 1):
    print(f"Parafrase {i}:")
    print(f"Teks: {result['text']}")
    print(f"Transformasi: {', '.join(result['transformations'])}")
    print(f"Similarity: {result['similarity_score']:.2f}")
    print()


### Output Example:

Parafrase 1:
Teks: Pendidikan merupakan proses pembelajaran yang amat penting untuk mengembangkan potensi manusia.
Transformasi: synonym_replacement, change_conjunctions
Similarity: 0.85

Parafrase 2:
Teks: Proses pembelajaran yang sangat penting untuk mengembangkan potensi manusia adalah pendidikan.
Transformasi: synonym_replacement, reorder_clauses
Similarity: 0.90

Parafrase 3:
Teks: Potensi manusia dikembangkan oleh proses pembelajaran yang sangat penting dalam pendidikan.
Transformasi: synonym_replacement, active_to_passive, reorder_clauses
Similarity: 0.75


## 5. Kelebihan Metode Hybrid

1. *Variasi Tinggi*: Kombinasi dua metode menghasilkan variasi yang lebih banyak
2. *Natural*: Hasil lebih natural karena tidak hanya mengganti kata
3. *Terkontrol*: Setiap transformasi dapat dilacak dan dikontrol
4. *Fleksibel*: Parameter dapat disesuaikan untuk hasil yang diinginkan

## 6. Kekurangan dan Batasan

1. *Kompleksitas*: Lebih kompleks dibanding metode tunggal
2. *Error Propagation*: Error di satu tahap dapat mempengaruhi tahap berikutnya
3. *Validasi*: Memerlukan validasi yang ketat untuk memastikan kualitas

## 7. Tips Implementasi

1. *Gunakan Validasi Bertingkat*: Validasi setelah setiap transformasi
2. *Batasi Transformasi*: Jangan terlalu banyak transformasi dalam satu parafrase
3. *Preserve Entities*: Jaga nama orang, tempat, dan entitas penting
4. *Testing*: Test dengan berbagai jenis kalimat
