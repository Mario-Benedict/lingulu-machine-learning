# Deployment Guide - Google Cloud Run

This document provides step-by-step instructions for deploying the Lingulu ML API to Google Cloud Run.

## 🌏 Region & GPU Support

**Region**: `asia-southeast1` (Singapore)

This deployment uses **Singapore region** because:
- ✅ **GPU support available** (for future GPU-accelerated inference)
- 🚀 Low latency for Southeast Asia users
- 🔧 Full Cloud Run features support

> **Note**: The Dockerfile is built with NVIDIA CUDA support. While Cloud Run currently runs on CPU, the same image can be deployed to GKE or Cloud Run with GPU when GPU instances become fully available.

## Prerequisites

1. **Google Cloud Account** with billing enabled
2. **Google Cloud Project** created
3. **Required APIs enabled**:
   - Cloud Run API
   - Artifact Registry API
   - Cloud Build API

4. **Required permissions**:
   - Cloud Run Admin
   - Artifact Registry Admin
   - Service Account User

## Setup Google Cloud

### 1. Create Artifact Registry Repository

```bash
# Set project ID
export PROJECT_ID="your-project-id"

# Create repository for Docker images in Singapore (GPU support)
gcloud artifacts repositories create lingulu \
    --repository-format=docker \
    --location=asia-southeast1 \
    --description="Lingulu ML API Docker images"
```

### 2. Create Service Account for GitHub Actions

```bash
# Create service account
gcloud iam service-accounts create github-actions \
    --display-name="GitHub Actions Deployment"

# Grant necessary permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/run.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/iam.serviceAccountUser"

# Create and download key
gcloud iam service-accounts keys create github-actions-key.json \
    --iam-account=github-actions@$PROJECT_ID.iam.gserviceaccount.com
```

## Configure GitHub Repository

### Required GitHub Secrets

Add the following secrets to your GitHub repository (Settings → Secrets and variables → Actions):

| Secret Name | Description | Example Value |
|-------------|-------------|---------------|
| `GCP_PROJECT_ID` | Your Google Cloud Project ID | `lingulu-production` |
| `GCP_SA_KEY` | Service account key JSON (entire content) | Contents of `github-actions-key.json` |
| `FLASK_ENV` | Flask environment | `production` |
| `FLASK_DEBUG` | Debug mode | `false` |
| `MODEL_ID` | HuggingFace model ID | `marx90/lingulu_wav2vec2_pronounciation_finetune` |
| `SAMPLING_RATE` | Audio sampling rate | `16000` |
| `MAX_AUDIO_LENGTH_SECONDS` | Max audio duration | `60` |
| `MAX_FILE_SIZE_MB` | Max file upload size | `10` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Setting Secrets

1. Go to GitHub repository → **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Add each secret from the table above
4. For `GCP_SA_KEY`, paste the **entire content** of `github-actions-key.json`

## Deployment Methods

### 1. Automatic Deployment (Recommended)

The CI/CD pipeline automatically deploys to Cloud Run when:
- Code is pushed to `main` branch
- All tests pass in the CI workflow

**Workflow:**
```
Push to main → CI Tests → CD Deploy to Cloud Run
```

### 2. Manual Deployment via GitHub Actions

1. Go to **Actions** tab in GitHub repository
2. Select **CD - Deploy to Google Cloud Run**
3. Click **Run workflow**
4. Select branch `main`
5. Click **Run workflow** button

### 3. Manual Deployment via CLI

```bash
# From project root directory
export PROJECT_ID="your-project-id"
export REGION="asia-southeast1"  # Singapore region for GPU support
export SERVICE_NAME="lingulu-ml-api"

# Build and push image
docker build -t asia-southeast1-docker.pkg.dev/$PROJECT_ID/lingulu/$SERVICE_NAME:latest .
docker push asia-southeast1-docker.pkg.dev/$PROJECT_ID/lingulu/$SERVICE_NAME:latest

# Deploy to Cloud Run with GPU support
gcloud run deploy $SERVICE_NAME \
  --image asia-southeast1-docker.pkg.dev/$PROJECT_ID/lingulu/$SERVICE_NAME:latest \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --concurrency 80 \
  --min-instances 0 \
  --max-instances 10 \
  --set-env-vars "FLASK_ENV=production" \
  --set-env-vars "MODEL_ID=marx90/lingulu_wav2vec2_pronounciation_finetune" \
  --set-env-vars "SAMPLING_RATE=16000" \
  --set-env-vars "MAX_AUDIO_LENGTH_SECONDS=60" \
  --set-env-vars "MAX_FILE_SIZE_MB=10" \
  --set-env-vars "LOG_LEVEL=INFO"

# Note: For GPU instances, add --gpu flag when GPU support is available in your region
# gcloud run deploy $SERVICE_NAME --gpu 1 --gpu-type nvidia-tesla-t4
```

## Local Testing with Docker

### Build and Run Locally

```bash
# Build Docker image
docker build -t lingulu-ml-api .

# Run container
docker run -p 8080:8080 \
  -e PORT=8080 \
  -e FLASK_ENV=development \
  -e MODEL_ID=marx90/lingulu_wav2vec2_pronounciation_finetune \
  lingulu-ml-api
```

### Using Docker Compose (with GPU support)

```bash
# Build and start services
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Environment Variables Configuration

Environment variables are injected at deployment time via Cloud Run configuration.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | No | `8080` | Server port (Cloud Run sets this automatically) |
| `FLASK_ENV` | Yes | - | Flask environment (production/development) |
| `FLASK_DEBUG` | No | `false` | Enable debug mode |
| `MODEL_ID` | Yes | - | HuggingFace model identifier |
| `SAMPLING_RATE` | Yes | `16000` | Audio sampling rate in Hz |
| `MAX_AUDIO_LENGTH_SECONDS` | No | `60` | Maximum audio duration |
| `MAX_FILE_SIZE_MB` | No | `10` | Maximum upload file size |
| `LOG_LEVEL` | No | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR) |

### Updating Environment Variables

#### Via Cloud Run Console:
1. Go to Cloud Run → Select service → **Edit & Deploy New Revision**
2. Go to **Variables & Secrets** tab
3. Update environment variables
4. Click **Deploy**

#### Via CLI:
```bash
gcloud run services update lingulu-ml-api \
  --region asia-southeast1 \
  --set-env-vars "LOG_LEVEL=DEBUG"
```

## Monitoring & Troubleshooting

### View Logs

```bash
# Real-time logs
gcloud run services logs read lingulu-ml-api \
  --region asia-southeast1 \
  --limit 50 \
  --follow

# Filter errors
gcloud run services logs read lingulu-ml-api \
  --region asia-southeast1 \
  --log-filter "severity>=ERROR"
```

### Check Service Status

```bash
# Get service details
gcloud run services describe lingulu-ml-api \
  --region asia-southeast1

# Get service URL
gcloud run services describe lingulu-ml-api \
  --region asia-southeast1 \
  --format 'value(status.url)'
```

### Health Check

```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe lingulu-ml-api \
  --region asia-southeast1 \
  --format 'value(status.url)')

# Test health endpoint
curl $SERVICE_URL/api/model/health
```

## Cost Optimization

### Cloud Run Pricing

- **CPU**: Only charged when processing requests
- **Memory**: Only charged during request processing
- **Request**: $0.40 per million requests
- **CPU-time**: $0.00002 / vCPU-second
- **Memory-time**: $0.0000025 / GiB-second

### Optimization Tips

1. **Min instances**: Set to 0 to scale to zero when idle
2. **Max instances**: Limit based on expected load
3. **CPU allocation**: Set to "CPU is only allocated during request processing"
4. **Concurrency**: Increase to handle more requests per instance (80-100)

```bash
gcloud run services update lingulu-ml-api \
  --region asia-southeast1 \
  --cpu-throttling \
  --min-instances 0 \
  --max-instances 5 \
  --concurrency 100
```

## Security Best Practices

### 1. Authentication

For production with authentication:

```bash
gcloud run services update lingulu-ml-api \
  --region asia-southeast1 \
  --no-allow-unauthenticated
```

Then invoke with authentication:
```bash
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  $SERVICE_URL/api/model/health
```

### 2. VPC Connector

For private network access:

```bash
gcloud run services update lingulu-ml-api \
  --region asia-southeast1 \
  --vpc-connector your-vpc-connector
```

### 3. Secret Management

Use Google Secret Manager for sensitive data:

```bash
# Create secret
echo -n "sensitive-value" | gcloud secrets create model-api-key --data-file=-

# Grant access to Cloud Run service account
gcloud secrets add-iam-policy-binding model-api-key \
  --member="serviceAccount:lingulu-ml-api@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

# Deploy with secret
gcloud run services update lingulu-ml-api \
  --region asia-southeast2 \
  --update-secrets="API_KEY=model-api-key:latest"
```

## Rollback

### Rollback to Previous Revision

```bash
# List revisions
gcloud run revisions list \
  --service lingulu-ml-api \
  --region asia-southeast1

# Rollback to specific revision
gcloud run services update-traffic lingulu-ml-api \
  --region asia-southeast1 \
  --to-revisions REVISION_NAME=100
```

## CI/CD Pipeline Details

The GitHub Actions workflow (`.github/workflows/cd.yml`) performs:

1. ✅ **Check CI Status** - Ensures tests pass before deployment
2. 🔐 **Authenticate to GCP** - Using service account key
3. 🔨 **Build Docker Image** - Multi-stage optimized build
4. 📤 **Push to Artifact Registry** - Tagged with commit SHA and latest
5. 🚀 **Deploy to Cloud Run** - With environment variables injected
6. ✅ **Verify Deployment** - Health check validation

## Support

For issues or questions:
- Check Cloud Run logs
- Review GitHub Actions workflow logs
- Verify all secrets are set correctly
- Ensure GCP APIs are enabled

## References

- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Artifact Registry Documentation](https://cloud.google.com/artifact-registry/docs)
- [GitHub Actions for GCP](https://github.com/google-github-actions)
