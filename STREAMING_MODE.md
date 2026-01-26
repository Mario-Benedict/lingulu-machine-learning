# Streaming Mode - No Download Required! 🚀

## Problem yang Dipecahkan

### ❌ Masalah Sebelumnya:
1. **FFmpeg Error**: Error saat decode audio dengan transformers default
2. **Storage Besar**: LibriSpeech 100h clean = ~6GB download
3. **Slow Loading**: Harus download semua data sebelum training

### ✅ Solusi Sekarang:
1. **No FFmpeg Needed**: Audio decode menggunakan soundfile + librosa
2. **Zero Storage**: Streaming langsung dari HuggingFace
3. **Faster Start**: Langsung training tanpa download penuh

## Cara Kerja

### Mode Streaming
```python
# Sebelumnya (download semua):
dataset = load_dataset("librispeech_asr", "clean", split="train.100")
# ❌ Download 6GB+

# Sekarang (streaming):
dataset = load_dataset("librispeech_asr", "clean", split="train.100", streaming=True)
# ✅ No download! Stream on-demand
```

### Audio Decoding

```python
# Audio datang sebagai bytes dari streaming
audio_bytes = batch["audio"]["bytes"]

# Decode dengan soundfile (no FFmpeg!)
import io
import soundfile as sf
audio_array, sr = sf.read(io.BytesIO(audio_bytes))

# Resample jika perlu dengan librosa
if sr != 16000:
    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
```

## Konfigurasi Dataset Size

Edit [model/data_loader.py](model/data_loader.py) untuk adjust berapa banyak samples:

```python
def load_librispeech_datasets():
    # Train dataset
    train_dataset = load_dataset(..., streaming=True)
    train_dataset = train_dataset.take(5000)  # ← Ubah angka ini
    
    # Eval dataset
    eval_dataset = load_dataset(..., streaming=True)
    eval_dataset = eval_dataset.take(500)    # ← Ubah angka ini
    
    # Test dataset
    test_dataset = load_dataset(..., streaming=True)
    test_dataset = test_dataset.take(500)    # ← Ubah angka ini
```

### Rekomendasi Sample Counts:

| Use Case | Train | Eval | Test | Training Time | Accuracy |
|----------|-------|------|------|---------------|----------|
| **Quick Test** | 1000 | 100 | 100 | ~30 min | Low |
| **Development** | 5000 | 500 | 500 | ~2-3 hours | Medium |
| **Production** | All (~28K) | All (~2.7K) | All (~2.6K) | ~10-12 hours | High |

### Untuk Full Dataset:
```python
# Hapus .take() untuk gunakan semua data
train_dataset = load_dataset(..., streaming=True)
# train_dataset = train_dataset.take(5000)  # ← Comment atau hapus line ini
```

## Keuntungan Streaming Mode

### 1. **Zero Disk Usage** 💾
```bash
# Before: 
# ~/.cache/huggingface/datasets/librispeech_asr/
#   ├── train.100/  (6GB)
#   ├── dev/        (300MB)
#   └── test/       (300MB)
# Total: ~6.5GB

# After: 
# No files downloaded! 0GB
```

### 2. **Faster Startup** ⚡
```
Before:
  Download: 20-30 minutes
  Process:  5 minutes
  Total:    25-35 minutes

After:
  Download: 0 minutes
  Stream:   Start immediately
  Total:    0 minutes to start
```

### 3. **On-Demand Loading** 📡
- Data dimuat saat dibutuhkan
- Perfect untuk experimentation
- Tidak waste storage untuk data yang tidak dipakai

## Trade-offs

### Streaming Mode:
- ✅ No download time
- ✅ No disk space used
- ✅ Start training immediately
- ⚠️ Slightly slower per-batch (network latency)
- ⚠️ Requires internet during training
- ⚠️ Can't shuffle full dataset

### Download Mode:
- ✅ Faster per-batch (local disk)
- ✅ Can train offline
- ✅ Full dataset shuffling
- ❌ Long initial download
- ❌ Uses lots of disk space
- ❌ Wait before starting

## Troubleshooting

### Error: "Audio bytes not found"
```python
# Check audio format in dataset
for batch in dataset:
    print(batch["audio"].keys())
    break

# Should see: dict_keys(['bytes', 'path', 'sampling_rate'])
```

### Error: "Connection timeout"
```python
# Reduce batch size untuk mengurangi network load
batch_size: int = 2
```

### Slow training?
```python
# Option 1: Download mode untuk dataset kecil
dataset = load_dataset(..., streaming=False)

# Option 2: Increase workers (if enough bandwidth)
# Di training script, bisa set num_workers di DataLoader
```

### Ingin combine streaming + local cache?
```python
from datasets import load_dataset

# Download subset kemudian cache
dataset = load_dataset(..., streaming=True)
dataset = dataset.take(5000)

# Convert to local
local_dataset = Dataset.from_dict({
    k: [item[k] for item in dataset]
    for k in next(iter(dataset)).keys()
})

# Save locally
local_dataset.save_to_disk("./local_cache")

# Next time, load from disk
dataset = Dataset.load_from_disk("./local_cache")
```

## Best Practices

### 1. Development Cycle
```python
# Step 1: Quick test dengan streaming (100-1000 samples)
train_dataset = train_dataset.take(1000)

# Step 2: Development dengan streaming (5000 samples)  
train_dataset = train_dataset.take(5000)

# Step 3: Production - download full dataset
dataset = load_dataset(..., streaming=False)  # Download once
```

### 2. Network Optimization
```python
# Prefetch data untuk smooth training
dataset = dataset.prefetch(100)

# Batch loading untuk efficiency
for batch in dataset.iter(batch_size=32):
    # Process batch
```

### 3. Hybrid Approach
```python
# Stream large train dataset
train = load_dataset(..., split="train", streaming=True)

# Download small eval/test sets
eval = load_dataset(..., split="validation", streaming=False)
test = load_dataset(..., split="test", streaming=False)
```

## Monitoring

Check data usage during training:
```python
import psutil

def monitor_data_usage():
    net = psutil.net_io_counters()
    print(f"Downloaded: {net.bytes_recv / 1e9:.2f} GB")
    print(f"Uploaded: {net.bytes_sent / 1e9:.2f} GB")

# Call periodically during training
```

## Summary

| Feature | Streaming | Download |
|---------|-----------|----------|
| Storage | 0GB | ~6.5GB |
| Setup Time | 0 min | ~30 min |
| Training Speed | Slightly slower | Faster |
| Internet Required | Yes | No (after download) |
| Best For | Development, Quick experiments | Production, Final training |

**Recommendation**: 
- 🧪 **Development**: Use streaming dengan limited samples
- 🚀 **Production**: Download full dataset untuk final training
