# 🐯 Lingulu — Pronunciation Scoring Model (Machine Learning)

This module contains the Machine Learning pipeline used in **Lingulu** to evaluate English pronunciation using speech recognition and phoneme-level scoring.

The model is based on **Wav2Vec2**, fine-tuned for pronunciation assessment using a **phoneme vocabulary** and **GOP (Goodness of Pronunciation) scoring**.

---

## 🎯 Purpose

The goal of this model is **not** to transcribe speech into text, but to:

> Evaluate how accurately a user pronounces words and sentences at the phoneme level.

This enables Lingulu to provide **objective pronunciation feedback** to learners.

---

## 🧠 Core Concept

The pipeline works as follows:

1. User speaks a sentence
2. Audio is processed by **fine-tuned Wav2Vec2**
3. Model predicts **phoneme sequence probabilities**
4. GOP score is calculated per phoneme
5. Scores are aggregated into:
   - Phoneme score
   - Word score
   - Sentence pronunciation score
6. Results are sent to backend as evaluation feedback

---

## 🏗️ Model Architecture

| Component | Description |
|-----------|-------------|
| Base Model | moxeeeem/wav2vec2-finetuned-pronunciation-correction2 |
| Fine-tuning Target | Phoneme recognition (not text ASR) |
| Vocabulary | Custom phoneme set (ARPAbet/IPA-based) |
| Output | Phoneme probability distribution |
| Scoring Method | GOP (Goodness of Pronunciation) |

---
## 📂 Folder Structure
```
lingulu-machine-learning/
│── app/
│── notebooks/
│   ├── models/
│   │   ├── v1/
|   |   |  ├── model_finetune.ipynb
|   |   |  └── train_history.csv
│   │   ├── v2/
|   |   |  ├── model_finetune.ipynb
|   |   |  └── train_history.csv
│   │   └── v3/
|   |      ├── model_finetune.ipynb
|   |      └── train_history.csv
│   ├── audio_converter.ipynb
│   ├── dataset_sampling.ipynb
│   └── model_evaluate_v3.ipynb
│── .gitignore
└── requirements.txt
```
---
## ⚙️ Installation

### Requirements

- Python 3.11+
- PyTorch
- Transformers (HuggingFace)
- Librosa
- NumPy

### Install dependencies:

```bash
pip install -r requirements.txt
```

### Running server 
```bash
python -m app.app
```

## 🧮 GOP Scoring

GOP measures how closely a spoken phoneme matches the expected phoneme.

Formula : 

```lua
GOP(p) = log P(p | audio) - max log P(q | audio)
```
Where:
- p = expected phoneme
- q = all possible phonemes

Higher GOP = better pronunciation.

---
## 🔗 Integration with Backend

Input:

- Audio file from user

- Expected sentence

Output (JSON):

```json
{
    "audio_duration_seconds": 1.28,
    "audio_samples": 20480,
    "filename": "He is a teacher.mp3",
    "latency_seconds": 0.29187,
    "pronounciation_assessment": {
        "average_score": 57.7,
        "gop_latency_seconds": 0.19755,
        "text": "He is a teacher",
        "words": [
            {
                "phonemes": [
                    {
                        "phoneme": "h",
                        "score": 89.2
                    },
                    {
                        "phoneme": "i",
                        "score": 73.2
                    }
                ],
                "score": 81.2,
                "word": "He"
            },
            {
                "phonemes": [
                    {
                        "phoneme": "ɪ",
                        "score": 78.9
                    },
                    {
                        "phoneme": "z",
                        "score": 83.1
                    }
                ],
                "score": 81.0,
                "word": "is"
            },
            {
                "phonemes": [
                    {
                        "phoneme": "ə",
                        "score": 0.0
                    }
                ],
                "score": 0.0,
                "word": "a"
            },
            {
                "phonemes": [
                    {
                        "phoneme": "t",
                        "score": 100.0
                    },
                    {
                        "phoneme": "i",
                        "score": 95.5
                    },
                    {
                        "phoneme": "tʃ",
                        "score": 78.2
                    },
                    {
                        "phoneme": "ɚ",
                        "score": 0.0
                    }
                ],
                "score": 68.4,
                "word": "teacher"
            }
        ]
    },
    "reference_text": "He is a teacher",
    "status": "success",
    "transcription": "hiɪzʌtitʃɚ"
}
```

---

## 📊 Why Wav2Vec2?

Wav2Vec2 is used because:

- Strong representation of speech features

- Works well with limited labeled data

- Suitable for phoneme-level tasks

- State-of-the-art for speech understanding

---

## 🚀 Deployment

This API is deployed to **Google Cloud Run** in **Singapore region** (`asia-southeast1`) with **L4 GPU** for optimal performance.

### Architecture

- **Platform**: Google Cloud Run (Serverless container deployment)
- **Region**: Singapore (asia-southeast1) - GPU support available
- **GPU**: NVIDIA L4 (24GB VRAM) - **Required for best performance**
- **Container**: Docker with NVIDIA CUDA 12.1 runtime
- **Scaling**: Auto-scale 0→N instances based on traffic
- **Cost**: Pay-per-use (no charge when idle)

### ⚡ GPU Performance

**This project REQUIRES GPU for optimal performance:**

| Metric | CPU Only | **L4 GPU** | Improvement |
|--------|----------|------------|-------------|
| Inference Time | ~800ms | **~150ms** | 🚀 **5.3x faster** |
| Throughput | ~7 req/s | **~40 req/s** | 📈 **5.7x higher** |
| User Experience | Slow | **Fast** | ⭐ **Production-ready** |

**Recommended**: Deploy with **nvidia-l4** GPU for best price/performance ratio.

### Quick Deploy via GitHub Actions

The repository is configured with CI/CD pipeline:

1. **Push to `main` branch** → Automatically triggers:
   - CI: Run tests and validation
   - CD: Build Docker image → Push to Artifact Registry → Deploy to Cloud Run

2. **Manual Deploy**: 
   - Go to **Actions** tab → Select "CD - Deploy to Google Cloud Run" → Click "Run workflow"

### API Endpoints

Once deployed, the API is accessible at your Cloud Run service URL:

```
https://lingulu-ml-api-XXXXXX-as.a.run.app
```

**Available Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/model/health` | GET | Health check |
| `/api/model/predict` | POST | Pronunciation assessment |
| `/api/metrics` | GET | Performance metrics (p50, p90, p99) |

### Example Usage

```bash
# Health check
curl https://YOUR-SERVICE-URL/api/model/health

# Pronunciation assessment
curl -X POST https://YOUR-SERVICE-URL/api/model/predict \
  -F "file=@audio.wav" \
  -F "text=Hello world"
```

### Configuration

Environment variables are injected at deployment via Cloud Run configuration:

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_ID` | HuggingFace model ID | `marx90/lingulu_wav2vec2_pronounciation_finetune` |
| `SAMPLING_RATE` | Audio sampling rate | `16000` |
| `MAX_AUDIO_LENGTH_SECONDS` | Max audio duration | `60` |
| `MAX_FILE_SIZE_MB` | Max upload size | `10` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Local Testing

**CPU-only (slower):**
```bash
docker-compose up --build
```

**With GPU (requires NVIDIA Docker runtime):**
```bash
# Ensure nvidia-docker is installed
docker-compose up --build

# Verify GPU is detected
docker exec -it lingulu-ml-app nvidia-smi
```

API will be available at `http://localhost:5000`

```bash
curl http://localhost:5000/api/model/health
```

### Full Documentation

For detailed deployment instructions, environment setup, monitoring, and troubleshooting:

📖 **See [DEPLOYMENT.md](./DEPLOYMENT.md)** for complete guide

---

## 🌟 Future Improvements

- Support IPA phoneme set

- Noise-robust training

- Real-time scoring

- Accent adaptation

---

Made with love ❤️, lack of sleep 🥱 and tears 💧 by MACAN MULAZ 🐅
