---
title: Lingulu ML API
emoji: 🎤
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 5000
hardware: t4-medium
---

# Lingulu Machine Learning API

Production-ready pronunciation assessment API powered by Wav2Vec2.

## Features

- 🎯 Accurate pronunciation scoring using wav2vec2
- 📊 Detailed phoneme-level feedback with GOP (Goodness of Pronunciation)
- ⚡ GPU-accelerated inference
- 📈 Real-time metrics monitoring (p50, p90, p99 latency)
- 🔒 Production-grade error handling

## API Endpoints

### Health Check
```
GET /api/model/health
```

### Predict
```
POST /api/model/predict
Content-Type: multipart/form-data

Fields:
- file: audio file (wav, mp3, flac, ogg, m4a)
- text: reference text for pronunciation assessment (optional)
```

### Metrics
```
GET /api/metrics
```

Returns latency metrics including p50, p90, p99 percentiles.

## Usage Example

```python
import requests

# Upload audio for transcription
with open('audio.wav', 'rb') as f:
    response = requests.post(
        'https://YOUR-SPACE.hf.space/api/model/predict',
        files={'file': f}
    )
print(response.json())

# With pronunciation assessment
with open('audio.wav', 'rb') as f:
    response = requests.post(
        'https://YOUR-SPACE.hf.space/api/model/predict',
        files={'file': f},
        data={'text': 'hello world'}
    )
print(response.json())

# Check metrics
response = requests.get('https://YOUR-SPACE.hf.space/api/metrics')
print(response.json())
```

## Monitoring

The API tracks latency metrics that can be accessed via `/api/metrics` endpoint:
- `latency_p50_ms`: Median latency
- `latency_p90_ms`: 90th percentile latency
- `latency_p99_ms`: 99th percentile latency
- `total_requests`: Total number of requests processed
- `error_rate`: Percentage of failed requests

## Model

Uses fine-tuned Wav2Vec2 model: `marx90/lingulu_wav2vec2_pronounciation_finetune`

## Configuration

The API can be configured using environment variables. Configuration is automatically generated during deployment from GitHub Secrets.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_HOST` | `0.0.0.0` | Host to bind the Flask application |
| `FLASK_PORT` | `5000` | Port to run the application |
| `FLASK_ENV` | `production` | Environment mode (development/production/test) |
| `FLASK_DEBUG` | `False` | Enable debug mode |
| `MODEL_ID` | `marx90/lingulu_wav2vec2_pronounciation_finetune` | HuggingFace model ID |
| `SAMPLING_RATE` | `16000` | Audio sampling rate in Hz |
| `MAX_AUDIO_LENGTH_SECONDS` | `60` | Maximum audio length in seconds |
| `MAX_FILE_SIZE_MB` | `10` | Maximum file upload size in MB |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR) |

### How Configuration Works

This Space is deployed via GitHub Actions which automatically:
1. Reads configuration from GitHub Secrets
2. Generates `.env` file during deployment
3. Applies values (or uses defaults if not set)

To modify configuration:
1. Update GitHub Secrets in the source repository
2. Re-deploy (push changes or trigger manual deployment)

**Note:** Environment variables can also be overridden directly in Hugging Face Space Settings → Variables and secrets (requires space restart).

### Modifying Configuration

**For Repository Maintainers:**
Set GitHub Secrets in repository settings (see DEPLOYMENT.md).

**For Space Users:**
Fork the space or set variables in Space Settings.

## Technical Details

- **Framework**: Flask with Gunicorn
- **ML Model**: Wav2Vec2 with CTC head
- **Audio Processing**: librosa + soundfile
- **GPU Support**: CUDA 12.1 with cuDNN 8
- **Container**: Docker with NVIDIA runtime

---

Built with ❤️ by Lingulu Team
