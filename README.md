# Lingulu Machine Learning - Pronunciation Assessment Model

Model pronunciation assessment berbasis Wav2Vec2 dengan custom layers untuk evaluasi pronunciation menggunakan IPA phonemes.

## 🎯 Fitur Utama

- ✅ **Custom Architecture**: Pretrained Wav2Vec2 + 2 custom transformer layers
- ✅ **IPA Phonemes**: International Phonetic Alphabet untuk akurasi tinggi
- ✅ **Stress & Syllables**: Include stress markers dan syllable boundaries
- ✅ **Pronunciation Scoring**: Architecture siap untuk scoring (bukan hanya recognition)
- ✅ **Transfer Learning**: Memanfaatkan pretrained weights untuk efisiensi

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: Tidak perlu install FFmpeg! Audio decoding menggunakan soundfile + librosa.

### 2. Pilih Mode Dataset

#### Option A: Local Dataset (Recommended - Fastest)

Jika sudah punya LibriSpeech di folder lokal:

Edit `model/config.py`:
```python
use_local_dataset: bool = True
local_dataset_path: str = "C:/path/to/LibriSpeech"  # Update path!
```

Lihat [LOCAL_DATASET_GUIDE.md](LOCAL_DATASET_GUIDE.md) untuk detail.

#### Option B: Streaming (No Download)

```python
use_local_dataset: bool = False
use_streaming: bool = True
```

#### Option C: HuggingFace Download

```python
use_local_dataset: bool = False
use_streaming: bool = False
```

### 3. Train Model

```bash
cd model
python train.py
```

Model akan:
- ✅ Load dataset (local/streaming/download)
- ✅ Convert text ke IPA phonemes
- ✅ Train dengan custom architecture
- ✅ Save model di `./wav2vec2-pronunciation-ctc/`

### 4. Test Model

```bash
python example_usage.py
```

Edit `example_usage.py` untuk menggunakan audio file Anda sendiri.

## 📁 Struktur Project

```
lingulu-machine-learning/
├── model/
│   ├── config.py              # Konfigurasi model & training
│   ├── model.py               # Custom architecture dengan layer tambahan
│   ├── data_loader.py         # IPA phoneme conversion
│   ├── train.py               # Training script
│   ├── data_collator.py       # Data collation
│   ├── metrics.py             # Evaluation metrics
│   └── visualization.py       # Training visualization
├── requirements.txt           # Python dependencies
├── example_usage.py          # Contoh penggunaan model
├── PRONUNCIATION_UPGRADE.md  # Dokumentasi lengkap upgrade
└── README.md                 # File ini
```

## ⚙️ Konfigurasi

Edit `model/config.py` untuk mengubah:

### Dataset Settings
```python
# Option 1: Local Dataset (Fastest)
use_local_dataset: bool = True
local_dataset_path: str = "C:/path/to/LibriSpeech"

# Option 2: Streaming (No storage)
use_local_dataset: bool = False
use_streaming: bool = True

# Option 3: Download
use_local_dataset: bool = False
use_streaming: bool = False
```

**Comparison:**
- **Local**: Fastest, offline, need ~6.5GB storage
- **Streaming**: 0GB storage, butuh internet, slower
- **Download**: Fast, offline after download, ~6.5GB cache

### Model Architecture
```python
use_custom_head: bool = True              # Enable custom layers
num_pronunciation_layers: int = 2          # Jumlah layer tambahan
pronunciation_head_dim: int = 256          # Dimensi pronunciation head
```

### Phoneme Settings
```python
use_ipa_phonemes: bool = True              # Gunakan IPA (bukan G2P)
include_stress_markers: bool = True        # Include stress ˈˌ
include_syllable_boundaries: bool = True   # Include syllables .
```

### Training
```python
batch_size: int = 4
num_epochs: int = 10
learning_rate: float = 1e-4
warmup_steps: int = 500
```

## 🎓 Cara Kerja

### 1. Text → IPA Phonemes

**Input text:**
```
"hello world"
```

**IPA phonemes:**
```
həˈloʊ|wɜrld
```

Dengan:
- `ˈ` = primary stress
- `|` = word boundary
- Standard IPA characters untuk pronunciation

### 2. Model Architecture

```
Audio (16kHz)
    ↓
Wav2Vec2 Feature Extractor (frozen)
    ↓
Wav2Vec2 Transformer Encoder
    ↓
Custom Pronunciation Layers (x2) ← TAMBAHAN
    ↓
Pronunciation Head (512→256→128)
    ↓
CTC Output → Phonemes
```

### 3. Training Process

1. **Load pretrained** Wav2Vec2 (facebook/wav2vec2-base-960h)
2. **Freeze** feature extractor
3. **Add** 2 custom transformer layers
4. **Train** dengan CTC loss untuk phoneme recognition
5. **Fine-tune** pronunciation-specific features

## 📊 Use Cases

### 1. Pronunciation Assessment
Evaluasi kualitas pronunciation learner:
```python
predicted = model.predict(audio)
expected = text_to_ipa("hello")
score = calculate_similarity(predicted, expected)
```

### 2. Speech-to-Phoneme
Convert speech langsung ke phonemes:
```python
phonemes = model.predict(audio)  # → "həˈloʊ"
```

### 3. Phoneme-level Feedback
Identifikasi phoneme mana yang salah:
```python
compare_phonemes(predicted, expected)
# Output: phoneme 'ə' at position 2 is incorrect
```

## 🔧 Advanced Usage

### Custom Dataset

Untuk menggunakan dataset sendiri:

```python
from datasets import Dataset

# Prepare your data
data = {
    "audio": [...],  # List of audio arrays
    "text": [...]    # List of transcripts
}

dataset = Dataset.from_dict(data)

# Convert to phonemes
dataset = dataset.map(text_to_phoneme_chars)

# Train
trainer.train(dataset)
```

### Inference Only

Untuk inference tanpa training:

```python
from model.model import Wav2Vec2ForPronunciationAssessment

# Load model
model = Wav2Vec2ForPronunciationAssessment.from_pretrained(
    "path/to/saved/model"
)

# Predict
outputs = model(audio_input)
phonemes = processor.decode(outputs.logits)
```

## 📈 Performance

Model ini menggunakan:
- **Base model**: facebook/wav2vec2-base-960h (95M params)
- **Custom layers**: ~8M params tambahan
- **Total**: ~103M params
- **Trainable**: ~20M params (dengan frozen feature extractor)

## 🐛 Troubleshooting

### Error: IPA conversion failed
```
Solution: Model akan fallback ke G2P otomatis
```

### Error: CUDA out of memory
```python
# Kurangi batch size
batch_size: int = 2
gradient_accumulation_steps: int = 8
```

### Error: vocab.json not found
```
Solution: Run training sekali untuk generate vocab
```

## 📚 References

- [Wav2Vec2 Paper](https://arxiv.org/abs/2006.11477)
- [IPA](https://en.wikipedia.org/wiki/International_Phonetic_Alphabet)
- [HuggingFace Transformers](https://huggingface.co/transformers/)

## 📝 License

[Your License Here]

## 👥 Contributors

- Mario Benedict

---

**Dokumentasi lengkap**: Lihat [PRONUNCIATION_UPGRADE.md](PRONUNCIATION_UPGRADE.md)
