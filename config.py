"""
config.py — TwistedVoice configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
AGENTS_DIR = PROJECT_ROOT / "agents"
STATIC_DIR = PROJECT_ROOT / "static"

AGENTS_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# === OLLAMA ===
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_API_CHAT = f"{OLLAMA_URL}/api/chat"

# === TWISTEDCOLLAB (for RAG-enabled agents) ===
TWISTEDCOLLAB_URL = os.getenv("TWISTEDCOLLAB_URL", "http://localhost:8000")

# === STT — faster-whisper ===
# Model sizes: tiny | base | small | medium | large-v3
# Recommendation: "base" for speed, "small" for better accuracy
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")

# === SERVER ===
SERVER_HOST = os.getenv("TWISTEDVOICE_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("TWISTEDVOICE_PORT", "8010"))

# === LLM DEFAULTS (override per-agent in YAML) ===
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "ministral-3:14b")
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 500  # Keep voice responses concise — agents can override

# === TTS — Piper ===
# Voice models are downloaded automatically on first use into models/
# Browse voices at: https://huggingface.co/rhasspy/piper-voices
# Format: <lang>/<lang_region>/<speaker>/<quality>/<filename stem>
# DEFAULT_TTS_VOICE example values:
#   en_US-lessac-medium   (neutral US English, good quality/speed balance)
#   en_US-amy-medium      (female US English)
#   en_GB-alan-medium     (British English, male)
DEFAULT_TTS_VOICE = os.getenv("TTS_VOICE", "en_US-lessac-medium")
# Playback speed multiplier (1.0 = normal, 1.1 = slightly faster)
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))
TTS_MODELS_DIR = PROJECT_ROOT / "models"
TTS_MODELS_DIR.mkdir(exist_ok=True)
