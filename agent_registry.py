"""
agent_registry.py — Load and manage voice agent definitions from YAML files.

Each agent is a YAML file in the agents/ directory. Drop in a new YAML,
POST /api/agents/reload, and the agent is immediately available — no restart.

YAML schema (all fields except 'name' are optional):

  name: my_agent
  description: "What this agent does"
  model: ministral-3:14b
  temperature: 0.7
  max_tokens: 400        # ceiling; used as-is when dynamic_tokens=false
  min_tokens: 150        # floor when dynamic_tokens=true (short/factual queries)
  dynamic_tokens: false  # scale max_tokens by query complexity
  rag_enabled: false     # route through TwistedCollab (true) or direct Ollama (false)
  rag_url: "http://localhost:8000"
  use_rag: true          # (TC only) enable document retrieval
  use_web_search: false  # (TC only) enable web search
  search_scope:          # (TC only) which collections to search; omit = TC default
    reference_papers: true
    my_papers: true
    notes: false
  voice_lang: "en-US"                # BCP-47 tag (fallback for browser speechSynthesis)
  tts_voice: "en_US-lessac-medium"    # Piper voice model name for server TTS
  system_prompt: |
    You are ...
"""
import logging
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from config import AGENTS_DIR, DEFAULT_MODEL, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS, TWISTEDCOLLAB_URL, DEFAULT_TTS_VOICE

logger = logging.getLogger("AgentRegistry")


@dataclass
class AgentConfig:
    name: str
    description: str
    system_prompt: str
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    min_tokens: int = 150            # floor when dynamic_tokens=True
    dynamic_tokens: bool = False     # scale max_tokens by query complexity
    rag_enabled: bool = False        # route via TwistedCollab (True) or direct Ollama
    rag_url: str = TWISTEDCOLLAB_URL
    use_rag: bool = True             # (TC) enable document retrieval
    use_web_search: bool = False     # (TC) enable web search
    search_scope: Optional[Dict[str, bool]] = None  # None = TC default
    voice_lang: str = "en-US"
    tts_voice: str = DEFAULT_TTS_VOICE

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "model": self.model,
            "rag_enabled": self.rag_enabled,
            "use_web_search": self.use_web_search,
            "dynamic_tokens": self.dynamic_tokens,
            "voice_lang": self.voice_lang,
            "tts_voice": self.tts_voice,
        }


class AgentRegistry:
    def __init__(self, agents_dir: Path = AGENTS_DIR):
        self.agents_dir = agents_dir
        self._agents: Dict[str, AgentConfig] = {}
        self.load()

    def load(self):
        """Scan agents directory and (re)load all YAML definitions."""
        self._agents = {}
        for yaml_file in sorted(self.agents_dir.glob("*.yaml")):
            try:
                with open(yaml_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if not data or "name" not in data:
                    logger.warning(f"Skipping {yaml_file.name}: missing 'name' field")
                    continue
                agent = AgentConfig(
                    name=data["name"],
                    description=data.get("description", ""),
                    system_prompt=data.get("system_prompt", "You are a helpful assistant."),
                    model=data.get("model", DEFAULT_MODEL),
                    temperature=float(data.get("temperature", DEFAULT_TEMPERATURE)),
                    max_tokens=int(data.get("max_tokens", DEFAULT_MAX_TOKENS)),
                    min_tokens=int(data.get("min_tokens", 150)),
                    dynamic_tokens=bool(data.get("dynamic_tokens", False)),
                    rag_enabled=bool(data.get("rag_enabled", False)),
                    rag_url=data.get("rag_url", TWISTEDCOLLAB_URL),
                    use_rag=bool(data.get("use_rag", True)),
                    use_web_search=bool(data.get("use_web_search", False)),
                    search_scope=data.get("search_scope", None) or None,
                    voice_lang=data.get("voice_lang", "en-US"),
                    tts_voice=data.get("tts_voice", DEFAULT_TTS_VOICE),
                )
                self._agents[agent.name] = agent
                logger.info(f"Loaded agent: '{agent.name}' (rag={agent.rag_enabled}, model={agent.model})")
            except Exception as e:
                logger.error(f"Failed to load {yaml_file.name}: {e}")
        logger.info(f"AgentRegistry: {len(self._agents)} agent(s) loaded.")

    def list(self) -> List[Dict]:
        return [a.to_dict() for a in self._agents.values()]

    def get(self, name: str) -> Optional[AgentConfig]:
        return self._agents.get(name)

    @property
    def default(self) -> Optional["AgentConfig"]:
        """Return the first loaded agent, or None if no agents are configured."""
        return next(iter(self._agents.values()), None)

    @property
    def default(self) -> Optional[AgentConfig]:
        """Return first agent as default (alphabetical by filename)."""
        return next(iter(self._agents.values()), None)
