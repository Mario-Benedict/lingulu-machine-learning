# Pronunciation Assessment Model - Upgrade Documentation

## Ringkasan Perubahan

Model telah diupgrade dari model phoneme recognition sederhana menjadi **pronunciation assessment model** yang lebih canggih dengan fitur-fitur berikut:

### 1. **Custom Architecture dengan Layer Tambahan**

Model sekarang menggunakan arsitektur custom `Wav2Vec2ForPronunciationAssessment` yang menambahkan:

- **Pretrained Wav2Vec2 base model** (facebook/wav2vec2-base-960h) sebagai feature extractor
- **2 layer Transformer tambahan** khusus untuk pronunciation assessment
- **Pronunciation scoring head** dengan 2-layer feed-forward network
- **Custom CTC loss** untuk phoneme recognition

#### Arsitektur Model:
```
Input Audio (16kHz)
    ↓
Wav2Vec2 Feature Extractor (frozen)
    ↓
Wav2Vec2 Encoder (pretrained, fine-tunable)
    ↓
Custom Transformer Layers (x2) ← LAYER TAMBAHAN
    ↓
Pronunciation Head (512 → 256 → 128)
    ↓
CTC Output Layer → Phonemes
```

### 2. **IPA Phoneme Conversion**

System sekarang menggunakan **IPA (International Phonetic Alphabet)** untuk konversi text-to-phoneme yang lebih akurat:

**Sebelumnya (G2P):**
- "hello world" → "hheelllloowwoorrlldd"

**Sekarang (IPA):**
- "hello world" → "həˈloʊ|wɜrld" (dengan stress markers dan word boundaries)

#### Fitur IPA:
- ✅ **Stress markers**: ˈ (primary stress), ˌ (secondary stress)
- ✅ **Syllable boundaries**: . (syllable separator)
- ✅ **Word boundaries**: | (word separator)
- ✅ **Accurate pronunciation**: Menggunakan standard IPA untuk konsistensi

### 3. **Configuration yang Fleksibel**

File `config.py` sekarang memiliki parameter-parameter baru:

```python
# Custom pronunciation layers
use_custom_head: bool = True              # Enable/disable custom layers
num_pronunciation_layers: int = 2          # Jumlah layer tambahan
pronunciation_head_dim: int = 256          # Dimensi pronunciation head

# Phoneme configuration
use_ipa_phonemes: bool = True              # Gunakan IPA (not G2P)
include_stress_markers: bool = True        # Include stress patterns
include_syllable_boundaries: bool = True   # Mark syllable boundaries

# Training improvements
warmup_steps: int = 500                    # Learning rate warmup
```

## Cara Menggunakan

### 1. Install Dependencies Baru

```bash
pip install -r requirements.txt
```

Library baru yang ditambahkan:
- `eng-to-ipa`: Untuk konversi text ke IPA phonemes

### 2. Training Model

```bash
python model/train.py
```

Model akan:
1. Load pretrained Wav2Vec2
2. Add 2 custom transformer layers
3. Add pronunciation scoring head
4. Train dengan IPA phonemes (bukan G2P biasa)

### 3. Konfigurasi Custom

Anda bisa mengubah behavior di `config.py`:

**Untuk menggunakan G2P biasa (bukan IPA):**
```python
use_ipa_phonemes: bool = False
```

**Untuk menambah layer tambahan:**
```python
num_pronunciation_layers: int = 3  # Default: 2
```

**Untuk disable custom head (gunakan model standard):**
```python
use_custom_head: bool = False
```

## Perbedaan dengan Model Sebelumnya

| Aspek | Sebelumnya | Sekarang |
|-------|-----------|----------|
| **Model** | Wav2Vec2ForCTC (standard) | Custom Wav2Vec2ForPronunciationAssessment |
| **Layers** | Pretrained saja | Pretrained + 2 custom transformer layers |
| **Phonemes** | G2P sederhana | IPA dengan stress & syllables |
| **Head** | Linear CTC head | Pronunciation scoring head + CTC |
| **Attention Mask** | Disabled | Enabled untuk custom layers |
| **Memory** | Standard | Gradient checkpointing enabled |

## Keuntungan Upgrade Ini

### 1. **Pronunciation yang Lebih Akurat**
- IPA memberikan representasi pronunciation yang standard dan akurat
- Stress markers membantu model belajar intonasi yang benar
- Syllable boundaries membantu segmentation

### 2. **Model yang Lebih Ekspresif**
- Custom transformer layers dapat belajar pronunciation-specific features
- Pronunciation head memberikan representasi intermediate yang berguna
- Bisa di-extend untuk pronunciation scoring (tidak hanya recognition)

### 3. **Fine-tuning yang Lebih Baik**
- Pretrained weights tetap digunakan (transfer learning)
- Custom layers hanya belajar pronunciation-specific features
- Lebih efisien daripada training from scratch

## Use Cases

Model ini cocok untuk:

1. **Pronunciation Assessment**: Menilai kualitas pronunciation
2. **Phoneme Recognition**: Mengenali phoneme dari audio
3. **Speech-to-Phoneme**: Convert speech langsung ke IPA phonemes
4. **Language Learning Apps**: Feedback untuk learners
5. **Accent Detection**: Menganalisis pronunciation patterns

## Troubleshooting

### Jika IPA conversion gagal:
Model akan fallback ke G2P otomatis. Check logs untuk error messages.

### Jika memory insufficient:
Kurangi batch size atau gunakan gradient accumulation:
```python
batch_size: int = 2
gradient_accumulation_steps: int = 8
```

### Jika ingin model lebih sederhana:
Set `use_custom_head = False` di config untuk menggunakan standard CTC head.

## Next Steps (Opsional)

Untuk development lebih lanjut, Anda bisa:

1. **Add pronunciation scoring**: Tambahkan regression head untuk scoring
2. **Multi-task learning**: Combine phoneme recognition + quality scoring
3. **Speaker adaptation**: Add speaker-specific layers
4. **Real-time inference**: Optimize untuk production

## References

- Wav2Vec2: https://arxiv.org/abs/2006.11477
- IPA: https://en.wikipedia.org/wiki/International_Phonetic_Alphabet
- CTC Loss: https://www.cs.toronto.edu/~graves/icml_2006.pdf
