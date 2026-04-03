"""
voice_manager.py — Speech-to-Text (faster-whisper) + Text-to-Speech (Piper).

Phase 1: STT only — browser handles TTS via window.speechSynthesis.
Phase 2: Full local TTS via Piper. Voice models auto-downloaded on first use.

Audio decoding strategy (STT):
  Browser MediaRecorder outputs different formats across devices —
    Chrome/Android : audio/webm;codecs=opus
    Firefox/Android: audio/ogg;codecs=opus
    Safari/iOS     : audio/mp4

  We use ffmpeg (subprocess) to decode any format to 16 kHz mono float32 PCM,
  which is exactly what faster-whisper expects.

TTS output:
  Piper synthesizes to 16-bit PCM WAV in memory. The WAV bytes are returned
  directly to the client which plays them via the HTML Audio API.
  Voice models (.onnx + .json) are auto-downloaded from Hugging Face on first
  use and cached in models/.
"""
import io
import logging
import subprocess
import urllib.request
import wave
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from faster_whisper import WhisperModel

from config import (
    WHISPER_MODEL_SIZE, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
    DEFAULT_TTS_VOICE, TTS_SPEED, TTS_MODELS_DIR,
)

logger = logging.getLogger("VoiceManager")

_SAMPLE_RATE = 16000  # faster-whisper requirement

# Piper voice model download base URL
_PIPER_HF_BASE = (
    "https://huggingface.co/rhasspy/piper-voices/resolve/main"
)


# ── Piper model auto-downloader ────────────────────────────────────────────────

def _piper_voice_url(voice_name: str) -> tuple[str, str]:
    """
    Derive the Hugging Face download URLs for a Piper voice name.

    voice_name format: <lang_REGION>-<speaker>-<quality>
    e.g. "en_US-lessac-medium"

    Returns (onnx_url, json_url).
    """
    parts = voice_name.split("-")
    lang_region = parts[0]                # e.g. en_US
    lang = lang_region.split("_")[0]      # e.g. en
    region = lang_region.split("_")[1] if "_" in lang_region else ""
    speaker = parts[1] if len(parts) > 1 else "default"
    quality = parts[2] if len(parts) > 2 else "medium"

    # HF repo path: <lang>/<lang_REGION>/<speaker>/<quality>/
    subpath = f"{lang}/{lang_region}/{speaker}/{quality}"
    stem = voice_name
    base = f"{_PIPER_HF_BASE}/{subpath}"
    return f"{base}/{stem}.onnx", f"{base}/{stem}.onnx.json"


def _ensure_piper_model(voice_name: str) -> tuple[Path, Path]:
    """Download Piper voice model files if not already cached. Returns (onnx, json) paths."""
    onnx_path = TTS_MODELS_DIR / f"{voice_name}.onnx"
    json_path = TTS_MODELS_DIR / f"{voice_name}.onnx.json"

    if not onnx_path.exists() or not json_path.exists():
        onnx_url, json_url = _piper_voice_url(voice_name)
        logger.info(f"Downloading Piper voice '{voice_name}'...")
        for url, path in [(onnx_url, onnx_path), (json_url, json_path)]:
            logger.info(f"  {url}")
            urllib.request.urlretrieve(url, path)
        logger.info(f"Piper voice '{voice_name}' cached to {TTS_MODELS_DIR}")
    return onnx_path, json_path


# ── Audio decode helper ────────────────────────────────────────────────────────

def _decode_audio(audio_bytes: bytes) -> np.ndarray:
    """
    Decode any audio format to 16 kHz mono float32 numpy array via ffmpeg.

    Args:
        audio_bytes: Raw audio bytes from the browser (any format).

    Returns:
        float32 numpy array of audio samples at 16 kHz.

    Raises:
        RuntimeError: If ffmpeg fails or produces no output.
    """
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(_SAMPLE_RATE),
        "pipe:1",
    ]
    result = subprocess.run(cmd, input=audio_bytes, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {result.stderr.decode()}")
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if len(audio) == 0:
        raise RuntimeError("ffmpeg returned empty audio — recording may be too short")
    return audio


# ── VoiceManager ──────────────────────────────────────────────────────────────

class VoiceManager:
    """STT via faster-whisper + TTS via Piper. Both load lazily on first call."""

    def __init__(self):
        self._stt_model: Optional[WhisperModel] = None
        self._tts_voices: Dict[str, object] = {}   # voice_name → PiperVoice instance
        logger.info(
            f"VoiceManager ready — STT '{WHISPER_MODEL_SIZE}' on {WHISPER_DEVICE}, "
            f"TTS voice '{DEFAULT_TTS_VOICE}' (both load on first use)"
        )

    # ── STT ──────────────────────────────────────────────────────────────────

    def _load_stt(self):
        if self._stt_model is None:
            device = WHISPER_DEVICE
            compute = WHISPER_COMPUTE_TYPE
            logger.info(f"Loading faster-whisper '{WHISPER_MODEL_SIZE}' ({device})...")
            try:
                self._stt_model = WhisperModel(
                    WHISPER_MODEL_SIZE, device=device, compute_type=compute,
                )
                logger.info(f"Whisper loaded on {device}.")
            except Exception as e:
                if device != "cpu":
                    logger.warning(f"Whisper GPU load failed ({e}). Falling back to CPU.")
                    self._stt_model = WhisperModel(
                        WHISPER_MODEL_SIZE, device="cpu", compute_type="int8",
                    )
                    logger.info("Whisper loaded on CPU (int8).")
                else:
                    raise

    def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcribe raw audio bytes → text string.

        Args:
            audio_bytes: Audio from browser MediaRecorder (any format; ffmpeg decodes it).

        Returns:
            Transcribed text. Empty string if no speech detected.
        """
        self._load_stt()
        audio = _decode_audio(audio_bytes)
        duration = len(audio) / _SAMPLE_RATE
        if duration < 0.3:
            logger.info("Audio too short to transcribe (<0.3s)")
            return ""
        segments, info = self._stt_model.transcribe(audio, beam_size=5)
        transcript = " ".join(seg.text.strip() for seg in segments).strip()
        logger.info(f"Transcribed {duration:.1f}s ({info.language}): {transcript!r}")
        return transcript

    # ── TTS ──────────────────────────────────────────────────────────────────

    def _load_tts(self, voice_name: str):
        if voice_name not in self._tts_voices:
            from piper.voice import PiperVoice
            onnx_path, _ = _ensure_piper_model(voice_name)
            logger.info(f"Loading Piper voice '{voice_name}'...")
            self._tts_voices[voice_name] = PiperVoice.load(str(onnx_path))
            logger.info(f"Piper voice '{voice_name}' loaded.")

    def synthesize(self, text: str, voice_name: Optional[str] = None) -> bytes:
        """
        Synthesize text to WAV audio bytes using Piper.

        Args:
            text:       Text to speak. Should be plain text (no markdown/HTML).
            voice_name: Piper voice name (e.g. 'en_US-lessac-medium').
                        Defaults to DEFAULT_TTS_VOICE from config.

        Returns:
            WAV file bytes suitable for direct HTTP response or HTML Audio playback.
        """
        from piper.config import SynthesisConfig
        voice_name = voice_name or DEFAULT_TTS_VOICE
        self._load_tts(voice_name)
        piper_voice = self._tts_voices[voice_name]

        syn_config = SynthesisConfig(length_scale=1.0 / TTS_SPEED)

        # Collect all audio chunks (one per sentence)
        chunks = list(piper_voice.synthesize(text, syn_config=syn_config))

        if not chunks:
            raise RuntimeError("Piper produced no audio output")

        sample_rate = chunks[0].sample_rate
        sample_width = chunks[0].sample_width
        sample_channels = chunks[0].sample_channels

        # Concatenate raw audio bytes from all chunks
        raw_audio = b""
        for chunk in chunks:
            # audio_float_array is float32 in [-1, 1]; convert to int16 PCM
            pcm = (chunk.audio_float_array * 32767).astype("int16")
            raw_audio += pcm.tobytes()

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            wav_file.setnchannels(sample_channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(raw_audio)

        wav_bytes = buf.getvalue()
        logger.info(
            f"Synthesized {len(text)} chars → {len(wav_bytes)//1024}KB WAV "
            f"(voice={voice_name})"
        )
        return wav_bytes

