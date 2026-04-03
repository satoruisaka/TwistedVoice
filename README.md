# TwistedVoice

A local, privacy-first voice assistant app. Talk to your AI agents by voice from any device on your network (or remotely via Tailscale). No cloud — everything runs on your own hardware.

## Features

- **Push-to-talk** voice interface optimised for mobile browsers
- **Server-side STT** via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (runs locally)
- **Server-side TTS** via [Piper](https://github.com/rhasspy/piper) (natural-sounding, runs locally)
- **Multiple agents** — each a YAML file with its own model, prompt, and capabilities
- **Web search** — live results via TwistedCollab (Brave API or DuckDuckGo fallback)
- **Document RAG** — semantic search over your personal document library via TwistedCollab
- **Dynamic token scaling** — short/factual queries get fast responses; complex queries get full detail
- **HTTPS** — self-signed cert for secure mic access from mobile browsers
- **Remote access** — works over [Tailscale](https://tailscale.com/)
- **Hot-reload** — drop in a new agent YAML and reload without restarting the server

## Architecture

```
Phone browser (push-to-talk)
       │
       ▼  HTTPS / Tailscale
TwistedVoice (FastAPI :8010)
  ├── POST /api/transcribe  ──► faster-whisper (STT)
  ├── POST /api/chat        ──► Ollama (direct)
  │                          └► TwistedCollab :8000 (RAG + web search)
  └── POST /api/tts         ──► Piper (TTS) → audio/wav → browser Audio API
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running on `localhost:11434`
- [ffmpeg](https://ffmpeg.org/) (for audio decoding)
- [TwistedCollab](../TwistedCollab/) running on `localhost:8000` (for RAG agents)

## Quick Start

```bash
cd TwistedVoice
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # edit as needed
```

Generate a self-signed TLS certificate (required for mic access on mobile):

```bash
openssl req -x509 -newkey rsa:4096 -sha256 -days 3650 -nodes \
  -keyout key.pem -out cert.pem \
  -subj "/CN=twistedvoice" \
  -addext "subjectAltName=IP:YOUR_SERVER_IP,IP:127.0.0.1"
```

Start the server:

```bash
../startTwistedVoice.sh
```

Open `https://YOUR_SERVER_IP:8010` on your phone. Accept the certificate warning on first visit.

## Configuration

All settings are in `config.py` and can be overridden via `.env`:

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `base` | faster-whisper model size (`tiny`, `base`, `small`, `medium`) |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `WHISPER_COMPUTE_TYPE` | `int8` | Whisper compute type (`int8`, `float16`, `float32`) |
| `DEFAULT_MODEL` | `ministral-3:14b` | Default Ollama model for new agents |
| `TTS_VOICE` | `en_US-lessac-medium` | Default Piper voice |
| `TTS_SPEED` | `1.0` | Speech rate multiplier |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `TWISTEDCOLLAB_URL` | `http://localhost:8000` | TwistedCollab base URL |

## Agents

Agents are YAML files in `agents/`. Add a file, then `POST /api/agents/reload` — no restart needed.

### Built-in agents

| Agent | Description |
|---|---|
| `quick_chat` | Fast conversational assistant — direct Ollama, no RAG |
| `web_search` | Live web search via TwistedCollab (Brave / DuckDuckGo) |
| `document_research` | Semantic search over your personal document library |
| `research_assistant` | General RAG assistant across all TwistedCollab collections |

### Agent YAML schema

```yaml
name: my_agent
description: "What this agent does"
model: ministral-3:14b
temperature: 0.7
max_tokens: 500          # token ceiling; used as-is when dynamic_tokens=false
min_tokens: 150          # floor when dynamic_tokens=true (short/factual queries)
dynamic_tokens: true     # scale max_tokens by query complexity
rag_enabled: false       # true = route through TwistedCollab
rag_url: "http://localhost:8000"
use_rag: true            # (TC) enable document retrieval
use_web_search: false    # (TC) enable web search
search_scope:            # (TC) which document collections to search
  reference_papers: true
  my_papers: true
  notes: false
voice_lang: "en-US"
tts_voice: "en_US-lessac-medium"
system_prompt: |
  You are ...
```

### Available TwistedCollab collections for `search_scope`

`reference_papers`, `my_papers`, `notes`, `user_uploads`, `news_articles`, `twistednews`, `skills`, `sessions`, `web_cache`, `debates`, `pics`, `dreams`

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Phone browser UI |
| GET | `/api/agents` | List available agents |
| GET | `/api/health` | Ollama + agent status |
| POST | `/api/transcribe` | Audio file → `{transcript}` |
| POST | `/api/chat` | `{agent_name, message, session_id}` → `{response}` |
| POST | `/api/tts` | `{text, agent_name}` → `audio/wav` |
| POST | `/api/chat/new` | `{}` → `{session_id}` |
| POST | `/api/agents/reload` | Hot-reload YAML files (no restart) |

## Voice Models

Piper voice models are downloaded automatically on first use into `models/`. To pre-download:

```bash
python -c "
from voice_manager import VoiceManager
VoiceManager().synthesize('test')
"
```

Browse available voices at [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices).

## Remote Access via Tailscale

1. Install Tailscale on server and client devices
2. Start TwistedVoice as usual
3. Access via your Tailscale IP: `https://<tailscale-ip>:8010`

No changes to server config needed.

## Privacy

All processing is local:
- STT: faster-whisper runs on your hardware
- LLM: Ollama runs on your hardware
- TTS: Piper runs on your hardware
- Web search: proxied through TwistedCollab (Brave API or DuckDuckGo)
- No telemetry, no external API calls for core features
