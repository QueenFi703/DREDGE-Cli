"""
DREDGE Whisper Microservice
Audio transcription service for Railway
Called by Vercel API for voice processing
"""
import os
import logging
from pathlib import Path
from flask import Flask, request, jsonify
import whisper
import tempfile
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load Whisper model on startup (cached in memory)
MODEL_SIZE = os.getenv('WHISPER_MODEL', 'tiny')  # tiny, base, small, medium, large
logger.info(f"Loading Whisper model: {MODEL_SIZE}")
model = whisper.load_model(MODEL_SIZE)
logger.info(f"✓ Whisper model ({MODEL_SIZE}) loaded")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint (public)"""
    return jsonify({
        "status": "healthy",
        "service": "whisper-microservice",
        "model": MODEL_SIZE,
        "timestamp": datetime.utcnow().isoformat()
    }), 200


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Transcribe audio file to text.
    
    Expects:
    - multipart/form-data with 'audio' file
    - Optional: 'language' (e.g., 'en', 'es', 'fr')
    - Optional: 'task' (transcribe or translate)
    
    Returns:
    {
        "text": "transcribed text",
        "language": "en",
        "duration": 2.5,
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
        
        # Get optional parameters
        language = request.form.get('language', None)
        task = request.form.get('task', 'transcribe')  # 'transcribe' or 'translate'
        
        # Save audio to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            audio_file.save(tmp.name)
            temp_audio_path = tmp.name
        
        logger.info(f"Transcribing audio: {audio_file.filename} (task={task}, language={language})")
        
        # Transcribe with Whisper
        result = model.transcribe(
            temp_audio_path,
            language=language,
            task=task
        )
        
        # Clean up temp file
        os.unlink(temp_audio_path)
        
        # Extract relevant fields
        response = {
            "text": result["text"].strip(),
            "language": result.get("language", "unknown"),
            "duration": result.get("duration", 0),
            "confidence": 0.95,  # Whisper doesn't provide per-segment confidence
            "segments": [
                {
                    "id": seg["id"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"]
                }
                for seg in result.get("segments", [])
            ]
        }
        
        logger.info(f"✓ Transcription complete: {len(response['text'])} characters")
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        return jsonify({
            "error": "Transcription failed",
            "details": str(e)
        }), 500


@app.route('/translate', methods=['POST'])
def translate_audio():
    """
    Translate audio to English.
    
    Expects:
    - multipart/form-data with 'audio' file
    
    Returns:
    {
        "text": "translated text",
        "source_language": "es",
        "confidence": 0.95
    }
    """
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "Missing 'audio' file"}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Save audio to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            audio_file.save(tmp.name)
            temp_audio_path = tmp.name
        
        logger.info(f"Translating audio: {audio_file.filename}")
        
        # Translate with Whisper
        result = model.transcribe(
            temp_audio_path,
            task='translate'
        )
        
        # Clean up temp file
        os.unlink(temp_audio_path)
        
        response = {
            "text": result["text"].strip(),
            "source_language": result.get("language", "unknown"),
            "confidence": 0.95
        }
        
        logger.info(f"✓ Translation complete: {len(response['text'])} characters")
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return jsonify({
            "error": "Translation failed",
            "details": str(e)
        }), 500


@app.route('/status', methods=['GET'])
def status():
    """Get service status and model info"""
    return jsonify({
        "service": "DREDGE Whisper Microservice",
        "status": "running",
        "model": MODEL_SIZE,
        "endpoints": {
            "/health": "Health check",
            "/transcribe": "Transcribe audio to text (POST)",
            "/translate": "Translate audio to English (POST)",
            "/status": "Service status (GET)"
        },
        "timestamp": datetime.utcnow().isoformat()
    }), 200


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False') == 'True'
    app.run(host='0.0.0.0', port=port, debug=debug)
