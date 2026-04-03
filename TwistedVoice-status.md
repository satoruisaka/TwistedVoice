# TwistedVoice — Build Status

## Phase 1 ✅ (browser push-to-talk, STT, Ollama chat, session history, HTTPS)
- FastAPI on port 8010, faster-whisper (CPU, base, int8), ministral-3:14b
- Self-signed cert for 192.168.1.92
- YAML agent system (quick_chat, research_assistant)
- In-memory session history, think-tag stripping (`<think>` blocks removed for reasoning models)

## Phase 2 ✅ (server-side TTS via Piper)
- piper-tts 1.4.2 installed in `.venv`
- Voice model: `en_US-lessac-medium` cached in `models/`
- `VoiceManager.synthesize()` uses `SynthesisConfig(length_scale=...)` + `AudioChunk` API
- `POST /api/tts` endpoint returns `audio/wav`
- `voice.html` `speak()` calls `/api/tts`, falls back to browser `speechSynthesis` on failure
- `tts_voice` field in `AgentConfig` and agent YAMLs for per-agent voice selection

## Phase 3 ✅ (Tailscale remote access)
- Tailscale installed on server and client
- TwistedVoice accessible from outside LAN via Tailscale
- No changes needed to server code — Tailscale handles the tunnel

## Phase 4 (planned): TwistedCollab-linked agents
- Specialized agents that surface TwistedCollab RAG data by domain/topic

---

## Key Files

| File | Purpose |
|---|---|
| `server.py` | FastAPI endpoints |
| `voice_manager.py` | STT (faster-whisper) + TTS (Piper) |
| `agent_registry.py` | YAML agent loader, `AgentConfig` |
| `agents/*.yaml` | Agent definitions (model, prompt, tts_voice, …) |
| `static/voice.html` | Phone-optimised push-to-talk UI |
| `config.py` | All configuration constants |
| `startTwistedVoice.sh` | Start script (uvicorn + SSL) |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Phone browser UI |
| GET | `/api/agents` | List available agents |
| GET | `/api/health` | Ollama + agent status |
| POST | `/api/transcribe` | Audio → `{transcript}` |
| POST | `/api/chat` | `{agent_name, message, session_id}` → `{response}` |
| POST | `/api/tts` | `{text, agent_name}` → `audio/wav` |
| POST | `/api/chat/new` | `{}` → `{session_id}` |
| POST | `/api/agents/reload` | Hot-reload YAMLs (no restart needed) |

---

## Hot-Reload Agents (no server restart)

```bash
curl -X POST https://localhost:8010/api/agents/reload -k
```

Change any YAML (model, prompt, tts_voice, …), then hit this endpoint.
Refresh the browser to update the agent dropdown.

---

## Configuration (`config.py` / `.env`)

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `base` | faster-whisper model size |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `WHISPER_COMPUTE_TYPE` | `int8` | Whisper compute type |
| `DEFAULT_MODEL` | `gemma4:e4b` | Ollama model for agents |
| `TTS_VOICE` | `en_US-lessac-medium` | Default Piper voice |
| `TTS_SPEED` | `1.0` | Speech rate (>1 = faster) |

---

## Starting the Server

```bash
cd /home/sator/project
./startTwistedVoice.sh
```

Then open `https://192.168.1.92:8010` on your phone (accept the self-signed cert warning).
