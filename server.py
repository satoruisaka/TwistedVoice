"""
server.py — TwistedVoice FastAPI server

Endpoints:
  GET  /                    → voice.html (phone browser UI)
  GET  /api/agents          → list available agents
  GET  /api/health          → Ollama + agent status
  POST /api/transcribe      → audio file → {transcript}
  POST /api/chat            → {agent_name, message} → {response, agent_name}
  POST /api/tts             → {text, agent_name} → audio/wav bytes (Piper TTS)
  POST /api/chat/new        → {} → {session_id}
  POST /api/agents/reload   → hot-reload YAML files from disk (no restart needed)
"""
import json
import logging
import re
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent_registry import AgentConfig, AgentRegistry
from config import OLLAMA_API_CHAT, OLLAMA_URL, STATIC_DIR, DEFAULT_TTS_VOICE
from voice_manager import VoiceManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("TwistedVoice")

app = FastAPI(title="TwistedVoice", version="1.0.0")

# Allow all origins so the phone on LAN can reach the server.
# For Phase 3 (Tailscale / public exposure) tighten this to specific origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global instances
registry = AgentRegistry()
voice = VoiceManager()

# ── Session history store ─────────────────────────────────────────────────────
# Keyed by session_id (str → list of {role, content} dicts).
# For RAG sessions, also tracks the TwistedCollab session_id.
# Lives in memory only — cleared on server restart (Phase 1 scope).
_history: Dict[str, List[dict]] = defaultdict(list)
_tc_sessions: Dict[str, str] = {}  # voice session_id → TwistedCollab session_id
MAX_HISTORY_TURNS = 20  # keep last N user+assistant pairs


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    agent_name: str
    message: str
    session_id: str = ""   # browser generates once per page load and reuses


class TTSRequest(BaseModel):
    text: str
    agent_name: str = ""      # optional — used to pick per-agent voice in future
    voice: str = ""           # override; empty = use DEFAULT_TTS_VOICE


class ChatResponse(BaseModel):
    response: str
    agent_name: str
    session_id: str


# ── LLM helpers ───────────────────────────────────────────────────────────────

_THINK_RE = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)

def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by reasoning models (e.g. gemma4)."""
    return _THINK_RE.sub('', text).strip()


# ── Speech-safe text cleaning ─────────────────────────────────────────────────

def _clean_for_speech(text: str) -> str:
    """
    Remove or neutralise symbols that TTS would read aloud literally.
    Applied to every response before it reaches Piper.
    """
    import re as _re

    # Remove HTML/XML tags
    text = _re.sub(r'<[^>]+>', '', text)

    # Markdown links [label](url) → label
    text = _re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Markdown images ![alt](url) → alt
    text = _re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)

    # Headers (## Title → Title)
    text = _re.sub(r'^#{1,6}\s+', '', text, flags=_re.MULTILINE)

    # Bold/italic: ***text***, **text**, *text*, ___text___, __text__, _text_
    text = _re.sub(r'\*{1,3}([^*\n]+)\*{1,3}', r'\1', text)
    text = _re.sub(r'_{1,3}([^_\n]+)_{1,3}', r'\1', text)

    # Inline code `code` and code fences ```...```
    text = _re.sub(r'```[\s\S]*?```', '', text)
    text = _re.sub(r'`([^`]+)`', r'\1', text)

    # Blockquotes (> text → text)
    text = _re.sub(r'^>\s?', '', text, flags=_re.MULTILINE)

    # Horizontal rules
    text = _re.sub(r'^[-*_]{3,}\s*$', '', text, flags=_re.MULTILINE)

    # Bullet/numbered list markers at line start
    text = _re.sub(r'^\s*[-*+•]\s+', '', text, flags=_re.MULTILINE)
    text = _re.sub(r'^\s*\d+[.)]\s+', '', text, flags=_re.MULTILINE)

    # Remaining stray symbols that TTS reads literally
    text = text.replace('\\', ' ')   # backslash
    text = text.replace('|', ', ')   # table pipes → pause
    text = text.replace('{', '').replace('}', '')
    text = text.replace('[', '').replace(']', '')
    text = text.replace('(', '').replace(')', '')
    text = text.replace('#', '')
    text = text.replace('@', ' at ')
    text = text.replace('&', ' and ')
    text = text.replace('%', ' percent ')
    text = text.replace('+', ' plus ')
    text = text.replace('=', ' equals ')
    text = text.replace('~', '')
    text = text.replace('^', '')
    text = _re.sub(r'\$(\d[\d,.]*)', r'\1 dollars', text)  # $42 → 42 dollars
    text = text.replace('$', ' dollars ')                   # bare $
    text = text.replace('>', '').replace('<', '')

    # Collapse multiple spaces / blank lines
    text = _re.sub(r'\n{2,}', '. ', text)
    text = _re.sub(r'\n', ' ', text)
    text = _re.sub(r'\s{2,}', ' ', text)

    return text.strip()


# Keywords that signal a detailed/long response is warranted
_LONG_QUERY_RE = re.compile(
    r'\b(summar\w+|explain|describe|detail\w*|elaborate|overview|analys\w+'
    r'|compar\w+|contrast|how does|how do|what are the|walk me through'
    r'|give me a|history of|tell me about|what is the difference)\b',
    re.IGNORECASE,
)

def _scale_tokens(message: str, agent: AgentConfig) -> int:
    """
    Return the max_tokens to use for this turn.

    When dynamic_tokens=True:
      - Long/detailed queries  → agent.max_tokens (configured ceiling)
      - Short/factual queries  → agent.min_tokens (configured floor)
    When dynamic_tokens=False:
      - Always agent.max_tokens
    """
    if not agent.dynamic_tokens:
        return agent.max_tokens
    return agent.max_tokens if _LONG_QUERY_RE.search(message) else agent.min_tokens


def _chat_ollama(agent: AgentConfig, message: str, history: List[dict], max_tokens: int) -> str:
    """Direct Ollama call with full conversation history."""
    messages = [{"role": "system", "content": agent.system_prompt}]
    messages.extend(history)   # prior turns
    messages.append({"role": "user", "content": message})
    payload = {
        "model": agent.model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": agent.temperature,
            "num_predict": max_tokens,
        },
    }
    try:
        resp = requests.post(OLLAMA_API_CHAT, json=payload, timeout=120)
        resp.raise_for_status()
        raw = resp.json()["message"]["content"]
        # Also check for thinking content in a separate field (newer Ollama builds)
        if not raw.strip():
            raw = resp.json()["message"].get("thinking", "")
        text = _strip_thinking(raw)
        return text if text else "I'm not sure how to respond to that."
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        raise HTTPException(status_code=502, detail=f"LLM error: {e}")


def _chat_twistedcollab(agent: AgentConfig, message: str, tc_session_id: str, max_tokens: int) -> str:
    """
    Call TwistedCollab's streaming chat endpoint and collect the full response.
    Reuses tc_session_id across turns so TwistedCollab maintains its own history.

    TwistedCollab returns Server-Sent Events with three event types:
      {"type": "content", "content": "<token>"}   — streamed token
      {"type": "replace", "content": "<full>"}    — post-distortion replacement
      {"type": "done", ...}                        — stream complete
    """
    payload = {
        "session_id": tc_session_id,
        "message": message,
        "model": agent.model,
        "temperature": agent.temperature,
        "max_tokens": max_tokens,
        "use_rag": agent.use_rag,
        "use_web_search": agent.use_web_search,
        "use_distortion": False,
    }
    if agent.search_scope:
        payload["search_scope"] = agent.search_scope
    url = f"{agent.rag_url}/api/chat/message/stream"
    try:
        resp = requests.post(url, json=payload, stream=True, timeout=300)
        resp.raise_for_status()

        content_parts = []
        distortion_content: Optional[str] = None
        tc_assigned_session: Optional[str] = None

        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data: "):
                continue
            try:
                ev = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            ev_type = ev.get("type")
            if ev_type == "token":
                content_parts.append(ev.get("content", ""))
            elif ev_type == "distortion":
                # TwistedCollab post-distortion content supersedes streamed tokens
                distortion_content = ev.get("content", "")
            elif ev_type == "session_id":
                tc_assigned_session = ev.get("session_id")
            elif ev_type == "error":
                logger.error(f"TwistedCollab error event: {ev.get('message', ev)}")
            elif ev_type == "done":
                break

        if tc_assigned_session:
            # Key by the tc_session_id argument so the caller can look up the real TC session
            _tc_sessions[tc_session_id] = tc_assigned_session

        return (distortion_content or "".join(content_parts)).strip()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TwistedCollab error: {e}")
        raise HTTPException(status_code=502, detail=f"TwistedCollab error: {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "voice.html"))


@app.get("/api/health")
async def health():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        ollama_ok = r.status_code == 200
    except Exception:
        ollama_ok = False
    return {
        "status": "ok",
        "ollama": "online" if ollama_ok else "offline",
        "agents": len(registry._agents),
    }


@app.get("/api/agents")
async def list_agents():
    return {"agents": registry.list()}


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Receive audio blob from browser MediaRecorder, return transcript."""
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")
    try:
        transcript = voice.transcribe(audio_bytes)
        return {"transcript": transcript}
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Route message to the selected agent, maintaining conversation history."""
    agent = registry.get(request.agent_name) or registry.default
    if agent is None:
        raise HTTPException(status_code=404, detail="No agents configured")

    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Empty message")

    # Assign or reuse session ID
    session_id = request.session_id.strip() or str(uuid.uuid4())
    history = _history[session_id]

    # Scale token budget based on query complexity (when agent opts in)
    max_tokens = _scale_tokens(message, agent)

    if agent.rag_enabled:
        # Reuse (or create) the paired TwistedCollab session so it maintains its own history
        tc_sid = _tc_sessions.get(session_id, "new")
        logger.info(f"[{agent.name}] RAG → TwistedCollab (tc_session={tc_sid}, max_tokens={max_tokens}): {message!r}")
        response = _chat_twistedcollab(agent, message, tc_sid, max_tokens)
        # After first call, _chat_twistedcollab stores "new" → tc_assigned_session.
        # Re-map our voice session_id to the real TC session_id for future turns.
        if tc_sid == "new" and "new" in _tc_sessions:
            _tc_sessions[session_id] = _tc_sessions.pop("new")
    else:
        logger.info(f"[{agent.name}] Direct Ollama (session={session_id}, max_tokens={max_tokens}): {message!r}")
        response = _chat_ollama(agent, message, history, max_tokens)

    # Update local history (used by direct Ollama path)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    # Trim to last MAX_HISTORY_TURNS pairs (2 messages per turn)
    if len(history) > MAX_HISTORY_TURNS * 2:
        _history[session_id] = history[-(MAX_HISTORY_TURNS * 2):]

    logger.info(f"[{agent.name}] Response ({len(response)} chars): {response[:80]!r}...")
    return ChatResponse(response=response, agent_name=agent.name, session_id=session_id)


@app.post("/api/chat/new")
async def new_session():
    """Generate a fresh session ID — call this when the user taps New Chat."""
    return {"session_id": str(uuid.uuid4())}


@app.post("/api/tts")
async def tts(request: TTSRequest):
    """Synthesize text to WAV audio using Piper. Returns audio/wav bytes."""
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")
    text = _clean_for_speech(text)
    # Preference order: explicit voice override → agent's tts_voice → default
    if request.voice.strip():
        voice_name = request.voice.strip()
    elif request.agent_name:
        agent = registry.get(request.agent_name)
        voice_name = agent.tts_voice if agent else DEFAULT_TTS_VOICE
    else:
        voice_name = DEFAULT_TTS_VOICE
    try:
        wav_bytes = voice.synthesize(text, voice_name)
        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")


@app.post("/api/agents/reload")
async def reload_agents():
    """Hot-reload agent YAML files — no server restart needed."""
    registry.load()
    return {"status": "ok", "agents": registry.list()}
