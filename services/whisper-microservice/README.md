# DREDGE Whisper Microservice

Audio transcription and translation service powered by OpenAI's Whisper model, deployed on Railway and called by Vercel.

## Features

- **Audio Transcription** — Convert speech to text
- **Audio Translation** — Translate speech to English
- **Multi-language Support** — Supports 99+ languages
- **Fast Inference** — Whisper tiny model (~39MB)
- **REST API** — Simple HTTP endpoints
- **Scalable** — Runs on Railway with auto-scaling

## Architecture

```
Vercel (api.dredge.ai)          Railway (whisper service)
    ↓ POST /api/audio/transcribe → /transcribe
    ↓ streams audio file        → processes with Whisper
    ← returns JSON              ← sends transcription
```

## Deployment

### Quick Start on Railway

1. **Connect Repository**
   ```bash
   railway login
   railway link
   ```

2. **Configure Environment**
   ```bash
   railway variables set WHISPER_MODEL=tiny
   railway variables set DEBUG=False
   ```

3. **Deploy**
   ```bash
   railway up
   ```

4. **Get Service URL**
   ```bash
   railway status
   # Note the public URL, e.g., https://whisper-production.up.railway.app
   ```

### Manual Docker Build

```bash
# Build
docker build -t whisper-service .

# Run locally
docker run -p 5000:5000 \
  -e WHISPER_MODEL=tiny \
  -e DEBUG=True \
  whisper-service

# Access
curl http://localhost:5000/health
```

## API Endpoints

### Health Check

```bash
GET /health

Response:
{
  "status": "healthy",
  "service": "whisper-microservice",
  "model": "tiny",
  "timestamp": "2026-07-07T12:00:00"
}
```

### Transcribe Audio

```bash
POST /transcribe
Content-Type: multipart/form-data

Fields:
  - audio: <audio file> (required)
  - language: en (optional, e.g., 'en', 'es', 'fr')
  - task: transcribe (optional, 'transcribe' or 'translate')

Response:
{
  "text": "Hello, this is a test",
  "language": "en",
  "duration": 2.5,
  "confidence": 0.95,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 2.5,
      "text": "Hello, this is a test"
    }
  ]
}
```

### Translate Audio

```bash
POST /translate
Content-Type: multipart/form-data

Fields:
  - audio: <audio file> (required)

Response:
{
  "text": "Hello, this is a test",
  "source_language": "es",
  "confidence": 0.95
}
```

### Service Status

```bash
GET /status

Response:
{
  "service": "DREDGE Whisper Microservice",
  "status": "running",
  "model": "tiny",
  "endpoints": { ... },
  "timestamp": "2026-07-07T12:00:00"
}
```

## Usage from Vercel

In your Vercel Flask app (`src/dredge/server.py` or new route):

```python
import requests
from flask import Blueprint, request, jsonify

audio_bp = Blueprint('audio', __name__, url_prefix='/api/audio')

# Get Whisper service URL from environment
WHISPER_SERVICE_URL = os.getenv(
    'WHISPER_SERVICE_URL',
    'https://whisper-production.up.railway.app'
)

@audio_bp.route('/transcribe', methods=['POST'])
@login_required
def transcribe_audio():
    """Forward audio to Whisper microservice"""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "Missing audio file"}), 400
        
        audio_file = request.files['audio']
        language = request.form.get('language', None)
        
        # Forward to Whisper service
        files = {'audio': (audio_file.filename, audio_file.stream)}
        data = {'language': language} if language else {}
        
        response = requests.post(
            f'{WHISPER_SERVICE_URL}/transcribe',
            files=files,
            data=data,
            timeout=300
        )
        
        return jsonify(response.json()), response.status_code
    
    except Exception as e:
        return jsonify({
            "error": "Transcription failed",
            "details": str(e)
        }), 500

@audio_bp.route('/translate', methods=['POST'])
@login_required
def translate_audio():
    """Translate audio via Whisper microservice"""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "Missing audio file"}), 400
        
        audio_file = request.files['audio']
        
        files = {'audio': (audio_file.filename, audio_file.stream)}
        response = requests.post(
            f'{WHISPER_SERVICE_URL}/translate',
            files=files,
            timeout=300
        )
        
        return jsonify(response.json()), response.status_code
    
    except Exception as e:
        return jsonify({
            "error": "Translation failed",
            "details": str(e)
        }), 500

app.register_blueprint(audio_bp)
```

## Environment Variables

| Variable | Default | Options | Description |
|----------|---------|---------|-------------|
| `WHISPER_MODEL` | `tiny` | tiny, base, small, medium, large | Model size (smaller = faster) |
| `DEBUG` | `False` | True, False | Debug mode |
| `PORT` | `5000` | Any port | Service port |

## Performance

| Model | Size | Memory | Speed | Accuracy |
|-------|------|--------|-------|----------|
| tiny | 39MB | ~500MB | Fast | ~95% |
| base | 140MB | ~1GB | Medium | ~97% |
| small | 466MB | ~1.5GB | Slower | ~98% |

## Testing

### Local Test

```bash
# Start service
python app.py

# Test transcribe (in another terminal)
curl -X POST http://localhost:5000/transcribe \
  -F "audio=@test_audio.wav"

# Test health
curl http://localhost:5000/health
```

### From Vercel (add to Vercel environment)

```bash
WHISPER_SERVICE_URL=https://whisper-production.up.railway.app
```

Then POST to your Vercel endpoint:
```bash
curl -X POST https://api.dredge.ai/api/audio/transcribe \
  -F "audio=@test_audio.wav" \
  -H "Authorization: Bearer <your_token>"
```

## Troubleshooting

### Model Download Timeout
Whisper downloads ~39MB on first run. If timeout:
```bash
# Pre-download model
railway run python -c "import whisper; whisper.load_model('tiny')"
```

### Memory Issues
If running out of memory, use `tiny` model:
```bash
railway variables set WHISPER_MODEL=tiny
```

### Slow Response
First request is slow (model loads). Subsequent requests are fast due to in-memory caching.

## Architecture Diagram

```
┌──────────────────────────┐
│   User / Frontend        │
│  (Browser or CLI)        │
└────────────┬─────────────┘
             │ audio file
             ▼
┌──────────────────────────┐
│   Vercel (api.dredge.ai) │
│  /api/audio/transcribe   │
│  (Gateway)               │
└────────────┬─────────────┘
             │ HTTP POST + audio
             ▼
┌──────────────────────────────────┐
│  Railway (whisper-service)       │
│  /transcribe or /translate       │
│  - Loads Whisper model (tiny)    │
│  - Processes audio (~2-5s)       │
│  - Returns JSON transcription    │
└────────────┬─────────────────────┘
             │ JSON response
             ▼
┌──────────────────────────┐
│   Vercel Response        │
│   (to user/frontend)     │
└──────────────────────────┘
```

## Next Steps

1. Deploy this service to Railway
2. Get the public URL
3. Add `WHISPER_SERVICE_URL` env var to Vercel
4. Add audio endpoints to Vercel app
5. Test end-to-end: Vercel → Railway → Whisper

## Support

For issues:
- Check Railway logs: `railway logs`
- Check Vercel logs: Vercel dashboard
- Test health: `curl https://whisper-service.up.railway.app/health`
