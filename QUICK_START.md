# Quick Reference - Streaming Mode

## ✅ Masalah Teratasi

### Error FFmpeg
```
❌ Error: ffmpeg not found or failed to decode audio
✅ Fixed: Menggunakan soundfile + librosa (no FFmpeg needed)
```

### Storage Penuh
```
❌ Before: Download 6.5GB dataset
✅ Now: Streaming mode = 0GB storage
```

## 🚀 Quick Commands

### Start Training (Streaming)
```bash
cd model
python train.py
```

### Change Sample Size
Edit `model/config.py`:
```python
# Quick test (fast)
streaming_train_samples: int = 1000

# Development (recommended)
streaming_train_samples: int = 5000

# Full dataset (slow but best accuracy)
streaming_train_samples: int = None  # or set use_streaming: bool = False
```

### Switch to Download Mode
Edit `model/config.py`:
```python
use_streaming: bool = False  # Download dataset sekali
```

## 📊 Performance Comparison

| Mode | Storage | Setup Time | Internet | Speed/Batch |
|------|---------|------------|----------|-------------|
| **Streaming (default)** | 0GB | 0 min | Required | Normal |
| **Download** | 6.5GB | ~30 min | Download only | Faster |

## 🔧 Configuration Presets

### Preset 1: Quick Test (Fastest)
```python
use_streaming: bool = True
streaming_train_samples: int = 1000
streaming_eval_samples: int = 100
streaming_test_samples: int = 100
batch_size: int = 8
num_epochs: int = 3
```
⏱️ Time: ~30 minutes

### Preset 2: Development (Balanced)
```python
use_streaming: bool = True
streaming_train_samples: int = 5000
streaming_eval_samples: int = 500
streaming_test_samples: int = 500
batch_size: int = 4
num_epochs: int = 10
```
⏱️ Time: ~2-3 hours

### Preset 3: Production (Best Quality)
```python
use_streaming: bool = False  # Download full dataset
batch_size: int = 4
num_epochs: int = 15
```
⏱️ Time: ~12-15 hours (including download)

## 💡 Tips

### Faster Training
```python
# Increase batch size (if you have GPU memory)
batch_size: int = 8
gradient_accumulation_steps: int = 2

# Reduce samples
streaming_train_samples: int = 3000
```

### Better Accuracy
```python
# Use more samples
streaming_train_samples: int = 10000
streaming_eval_samples: int = 1000

# Longer training
num_epochs: int = 15
```

### Save Internet Bandwidth
```python
# Download once, use forever
use_streaming: bool = False

# Or use smaller subset
streaming_train_samples: int = 2000
```

## 🐛 Common Issues

### Issue: Training too slow
**Solution**: Reduce samples or increase batch size
```python
streaming_train_samples: int = 2000
batch_size: int = 8
```

### Issue: Out of memory
**Solution**: Reduce batch size
```python
batch_size: int = 2
gradient_accumulation_steps: int = 8
```

### Issue: Want offline training
**Solution**: Download dataset
```python
use_streaming: bool = False
```

### Issue: Connection errors
**Solution**: Check internet or switch to download mode
```bash
# Download once
python -c "from datasets import load_dataset; load_dataset('librispeech_asr', 'clean', split='train.100')"
```

## 📁 File Locations

```
model/
├── config.py          ← Edit konfigurasi di sini
├── train.py           ← Run training
├── data_loader.py     ← Streaming logic
└── model.py           ← Model architecture

Generated files:
├── vocab.json         ← Vocabulary (auto-generated)
└── wav2vec2-pronunciation-ctc/
    └── final/         ← Trained model
```

## 🎯 Workflow

```
1. Edit config.py (set sample size)
   ↓
2. Run: python model/train.py
   ↓
3. Wait for training
   ↓
4. Model saved in ./wav2vec2-pronunciation-ctc/final/
   ↓
5. Test with: python example_usage.py
```

## 📚 More Info

- Full docs: [STREAMING_MODE.md](STREAMING_MODE.md)
- Architecture: [PRONUNCIATION_UPGRADE.md](PRONUNCIATION_UPGRADE.md)
- Main README: [README.md](README.md)
