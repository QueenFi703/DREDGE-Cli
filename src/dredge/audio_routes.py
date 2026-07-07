"""
Audio Processing Routes for Vercel
Proxies requests to Whisper microservice on Railway
"""
import os
import logging
from flask import Blueprint, request, jsonify
from flask_login import login_required
import requests

logger = logging.getLogger(__name__)

audio_bp = Blueprint('audio', __name__, url_prefix='/api/audio')

# Whisper service URL (set via environment variable)
WHISPER_SERVICE_URL = os.getenv(
    'WHISPER_SERVICE_URL',
    'https://whisper-production.up.railway.app'
)

logger.info(f"Audio routes configured with Whisper service: {WHISPER_SERVICE_URL}")


@audio_bp.route('/health', methods=['GET'])
def whisper_health():
    """Check Whisper service health"""
    try:
        response = requests.get(
            f'{WHISPER_SERVICE_URL}/health',
            timeout=5
        )
        return jsonify(response.json()), response.status_code
    except Exception as e:
        logger.error(f"Whisper service health check failed: {str(e)}")
        return jsonify({
            "status": "unreachable",
            "error": str(e)
        }), 503


@audio_bp.route('/transcribe', methods=['POST'])
@login_required
def transcribe_audio():
    """
    Transcribe audio file via Whisper service.
    
    Expected:
    - multipart/form-data with 'audio' file
    - Optional: 'language' (e.g., 'en', 'es')
    
    Returns:
    {
        "text": "transcribed text",
        "language": "en",
        "duration": 2.5,
        "confidence": 0.95,
        "segments": [...]
    }
    """
    try:
        # Validate request
        if 'audio' not in request.files:
            return jsonify({"error": "Missing 'audio' file"}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Get optional parameters
        language = request.form.get('language', None)
        
        logger.info(f"Transcribing audio: {audio_file.filename} (user: {request.headers.get('User-Agent')})")
        
        # Forward to Whisper service
        files = {'audio': (audio_file.filename, audio_file.stream, audio_file.content_type)}
        data = {}
        if language:
            data['language'] = language
        
        response = requests.post(
            f'{WHISPER_SERVICE_URL}/transcribe',
            files=files,
            data=data,
            timeout=300  # Transcription can take a while
        )
        
        if response.status_code == 200:
            logger.info(f"✓ Transcription successful")
        else:
            logger.error(f"Whisper service error: {response.status_code}")
        
        return jsonify(response.json()), response.status_code
    
    except requests.exceptions.Timeout:
        logger.error("Whisper service timeout")
        return jsonify({
            "error": "Transcription timeout",
            "details": "Service took too long to respond"
        }), 504
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        return jsonify({
            "error": "Transcription failed",
            "details": str(e)
        }), 500


@audio_bp.route('/translate', methods=['POST'])
@login_required
def translate_audio():
    """
    Translate audio to English via Whisper service.
    
    Expected:
    - multipart/form-data with 'audio' file
    
    Returns:
    {
        "text": "translated text in English",
        "source_language": "es",
        "confidence": 0.95
    }
    """
    try:
        # Validate request
        if 'audio' not in request.files:
            return jsonify({"error": "Missing 'audio' file"}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        logger.info(f"Translating audio: {audio_file.filename}")
        
        # Forward to Whisper service
        files = {'audio': (audio_file.filename, audio_file.stream, audio_file.content_type)}
        
        response = requests.post(
            f'{WHISPER_SERVICE_URL}/translate',
            files=files,
            timeout=300
        )
        
        if response.status_code == 200:
            logger.info(f"✓ Translation successful")
        else:
            logger.error(f"Whisper service error: {response.status_code}")
        
        return jsonify(response.json()), response.status_code
    
    except requests.exceptions.Timeout:
        logger.error("Whisper service timeout")
        return jsonify({
            "error": "Translation timeout",
            "details": "Service took too long to respond"
        }), 504
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return jsonify({
            "error": "Translation failed",
            "details": str(e)
        }), 500


@audio_bp.route('/status', methods=['GET'])
def audio_status():
    """Get audio service status"""
    try:
        response = requests.get(
            f'{WHISPER_SERVICE_URL}/status',
            timeout=5
        )
        return jsonify(response.json()), response.status_code
    except Exception as e:
        logger.error(f"Audio service status check failed: {str(e)}")
        return jsonify({
            "status": "unreachable",
            "error": str(e)
        }), 503
