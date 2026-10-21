import os
from loguru import logger
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
# Explicitly point to backend/.env to ensure it loads regardless of CWD
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class Settings:
    def __init__(self) -> None:
        # App
        self.APP_NAME: str = os.getenv("APP_NAME", "AI Surveillance API")
        self.VERSION: str = os.getenv("APP_VERSION", "0.1.0")
        self.DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
        # CORS
        default_origins = "http://localhost:3000,http://127.0.0.1:3000"
        origins_env = os.getenv("ALLOWED_ORIGINS", default_origins)
        self.ALLOWED_ORIGINS: List[str] = [o.strip() for o in origins_env.split(",") if o.strip()]

        # MongoDB
        # IMPORTANT: Do NOT hardcode credentials in code. Provide MONGODB_URI via environment variables.
        self.MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        self.MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "ai_surveillance")
        # Keep imports and offline evaluation usable when MongoDB is not running.
        # The previous driver defaults (20-30 s) made ``import backend.app.*``
        # appear to hang.  Fail fast and fall back to environment config.
        self.MONGO_SERVER_SELECTION_TIMEOUT_MS: int = int(
            os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", "800")
        )
        self._llm_config_cache: dict | None = None
        self._llm_config_checked: bool = False

        # Storage
        self.STORAGE_ROOT: Path = Path(os.getenv("STORAGE_ROOT", "data")).resolve()
        self.RECORDINGS_DIR: Path = self.STORAGE_ROOT / "recordings"
        self.CLIPS_DIR: Path = self.STORAGE_ROOT / "clips"
        self.SNAPSHOTS_DIR: Path = self.STORAGE_ROOT / "snapshots"
        self.LOGS_DIR: Path = self.STORAGE_ROOT / "logs"

        # Models (used by detection service; keep here for centralized config)
        self.MODEL_PATH: str = os.getenv("MODEL_PATH", "yolo11m-seg.pt")
        self.OPENVINO_MODEL_XML: str = os.getenv(
            "OPENVINO_MODEL_XML",
            "intel/person-attributes-recognition-crossroad-0238/FP32/person-attributes-recognition-crossroad-0238.xml",
        )

        # LLM / NLP (optional; wire later)
        # Providers: "opencode" (local opencode runtime, default), "openai" /
        # "openai_compatible" (alias "openai_compat": any OpenAI-compatible
        # endpoint), "openrouter", "ollama".  The opencode provider speaks the
        # OpenAI chat-completions contract and falls back to the opencode
        # session API and CLI.
        #
        # LLM_PROVIDER selects the transport; NL_DEFAULT_MODEL is the model
        # id of the opencode runtime only - the hosted providers have their
        # own defaults (see ``default_model_for``) so an opencode model id
        # can never be sent to the OpenAI API.
        self.LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "opencode").strip().lower()
        self.OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        # Base URL of any OpenAI-compatible endpoint (opencode serve, vLLM,
        # LM Studio, Ollama /v1, DeepSeek, ...).  Used by the openai_compat
        # provider and by the opencode HTTP transport.  Empty by default:
        # the opencode provider talks to OPENCODE_BASE_URL instead and the
        # hosted OpenAI endpoint is only used when it is configured.
        self.OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "").strip()
        # opencode runtime: server URL and model id.  The default model is the
        # free DeepSeek V4.1 Flash served by the local opencode runtime.
        self.OPENCODE_BASE_URL: str = os.getenv(
            "OPENCODE_BASE_URL", "http://127.0.0.1:4096"
        ).strip()
        # True when OPENCODE_BASE_URL was really configured (not the
        # built-in default).  The CLI is only pointed at a server that was
        # asked for explicitly.
        self.OPENCODE_BASE_URL_EXPLICIT: bool = bool(
            os.getenv("OPENCODE_BASE_URL", "").strip()
        )
        self.OPENCODE_MODEL: str = os.getenv(
            "OPENCODE_MODEL", "opencode-go/beam-free"
        ).strip()
        # Tried in order when the primary model is unavailable (e.g. the free
        # tier's usage limit is exhausted).  Only free models by default; a
        # paid model has to be added explicitly.
        self.OPENCODE_MODEL_FALLBACKS: str = os.getenv(
            "OPENCODE_MODEL_FALLBACKS", "opencode-go/minimax-m3.1-flash-free"
        ).strip()
        # auto | http | session | cli  (transport order for the opencode
        # provider).  "auto" expands to http -> session -> cli, so the CLI -
        # the transport that is known to work with the local runtime - stays
        # the last resort.
        self.OPENCODE_TRANSPORT: str = os.getenv("OPENCODE_TRANSPORT", "auto").strip().lower()
        # How long a transport that answered "route not available" is skipped
        # before it is probed again (seconds).
        self.OPENCODE_TRANSPORT_RETRY_SECONDS: int = int(
            os.getenv("OPENCODE_TRANSPORT_RETRY_SECONDS", "120")
        )
        self.OPENCODE_CLI: str = os.getenv("OPENCODE_CLI", "").strip()
        # Pass "--server <url>" to the opencode CLI.  Default off: the CLI
        # only talks to a runtime server when one was configured explicitly
        # (OPENCODE_BASE_URL in the environment) or this switch is on.
        self.OPENCODE_CLI_USE_SERVER: bool = os.getenv(
            "OPENCODE_CLI_USE_SERVER", "false"
        ).strip().lower() in ("1", "true", "yes", "on")
        self.OPENCODE_SERVER_PASSWORD: str = os.getenv("OPENCODE_SERVER_PASSWORD", "").strip()
        self.NL_MAX_TOKENS: int = int(os.getenv("NL_MAX_TOKENS", "1400"))
        # NL_TIMEOUT_SECONDS is the budget of one parse (all models and all
        # transports together); LLM_TIMEOUT overrides it.
        self.NL_TIMEOUT_SECONDS: int = int(os.getenv("NL_TIMEOUT_SECONDS", "300"))
        self.LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", str(self.NL_TIMEOUT_SECONDS)))

        # Auto-detect OpenRouter when the key starts with sk-or- and the
        # provider was not explicitly set to something else.
        if self.OPENAI_API_KEY.startswith("sk-or-") and self.LLM_PROVIDER in ("", "openai"):
            self.LLM_PROVIDER = "openrouter"

        self.OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.NL_DEFAULT_MODEL: str = os.getenv("NL_DEFAULT_MODEL", "opencode-go/beam-free")
        # Date used to ground relative expressions ("today", "yesterday").
        # Defaults to the released archive epoch so the demo does not resolve
        # those words against the machine clock.  Set DATA_EPOCH_DATE to a new
        # date for a fresh capture, or to an empty value to use "now".
        self.DATA_EPOCH_DATE: str = os.getenv(
            "DATA_EPOCH_DATE", "2026-03-01"
        ).strip()

        # Semantic VLM / Vector Search (optional; can be disabled)
        self.ENABLE_SEMANTIC: bool = os.getenv("ENABLE_SEMANTIC", "true").lower() == "true"
        # open_clip model id and pretrained tag
        # Example: "ViT-B-32/laion2b_s34b_b79k" or "ViT-B-32" + pretrained tag handled in code
        self.SEMANTIC_MODEL: str = os.getenv("SEMANTIC_MODEL", "ViT-B-32/laion2b_s34b_b79k")
        self.EMBED_DEVICE: str = os.getenv("EMBED_DEVICE", "cuda")  # "cuda" | "cpu"
        self.EMBED_BATCH_SIZE: int = int(os.getenv("EMBED_BATCH_SIZE", "32"))
        self.VECTOR_BACKEND: str = os.getenv("VECTOR_BACKEND", "faiss")  # "faiss"
        # FAISS storage (frame-level CLIP)
        self.FAISS_DIR: Path = Path(os.getenv("FAISS_DIR", str(self.STORAGE_ROOT / "index" / "faiss"))).resolve()
        # Person-level vector store (SigLIP+attributes fusion)
        self.FAISS_PERSON_DIR: Path = Path(os.getenv("FAISS_PERSON_DIR", str(self.STORAGE_ROOT / "index" / "faiss_person"))).resolve()
        # If using IVF later, you can tune nprobe; for Flat, it is not used
        self.FAISS_NPROBE: int = int(os.getenv("FAISS_NPROBE", "32"))

        # Multimodal person encoder defaults
        self.SIGLIP_MODEL: str = os.getenv("SIGLIP_MODEL", "google/siglip-base-patch16-224")
        self.TEXT_ENCODER_NAME: str = os.getenv("TEXT_ENCODER_NAME", "all-MiniLM-L12-v2")
        self.FUSION_VISUAL_WEIGHT: float = float(os.getenv("FUSION_VISUAL_WEIGHT", "0.6"))
        self.FUSION_TEXT_WEIGHT: float = float(os.getenv("FUSION_TEXT_WEIGHT", "0.4"))

        # Captions (BLIP-2)
        self.ENABLE_CAPTIONS: bool = os.getenv("ENABLE_CAPTIONS", "true").lower() == "true"
        self.CAPTION_MODEL: str = os.getenv("CAPTION_MODEL", "clip_labels")
        self.CAPTION_MAX_NEW_TOKENS: int = int(os.getenv("CAPTION_MAX_NEW_TOKENS", "50"))
        self.CAPTION_TOP_K: int = int(os.getenv("CAPTION_TOP_K", "3"))
        # Hugging Face auth for private/large models
        self.HF_TOKEN: str = os.getenv("HF_TOKEN", "")
        self.HF_HOME: str = os.getenv("HF_HOME", "")

        # Retrieval accuracy improvements (all default True; set false to disable)
        self.ENABLE_QUERY_EXPANSION: bool = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"
        self.ENABLE_ADAPTIVE_ALPHA: bool = os.getenv("ENABLE_ADAPTIVE_ALPHA", "true").lower() == "true"
        self.ENABLE_RECENCY_BOOST: bool = os.getenv("ENABLE_RECENCY_BOOST", "true").lower() == "true"
        self.ENABLE_ADAPTIVE_CONFIDENCE: bool = os.getenv("ENABLE_ADAPTIVE_CONFIDENCE", "true").lower() == "true"
        self.ENABLE_RESULT_DIVERSITY: bool = os.getenv("ENABLE_RESULT_DIVERSITY", "true").lower() == "true"
        self.ENABLE_ADAPTIVE_GAPS: bool = os.getenv("ENABLE_ADAPTIVE_GAPS", "true").lower() == "true"
        self.ENABLE_TEMPORAL_OVERLAP: bool = os.getenv("ENABLE_TEMPORAL_OVERLAP", "true").lower() == "true"
        self.ENABLE_CLIP_EXPANSION: bool = os.getenv("ENABLE_CLIP_EXPANSION", "true").lower() == "true"
        # Recency boost: half-life in hours (score multiplier decays with age)
        self.RECENCY_HALFLIFE_HOURS: float = float(os.getenv("RECENCY_HALFLIFE_HOURS", "24.0"))
        # MMR diversity: 0 = no diversity, 1 = max diversity
        self.MMR_DIVERSITY_LAMBDA: float = float(os.getenv("MMR_DIVERSITY_LAMBDA", "0.3"))

        # ------------------------------------------------------------------
        # Retrieval routing (deployment switch)
        # ------------------------------------------------------------------
        # When true, person/attribute/colour queries are resolved against the
        # 1152-d person FAISS index and the matching scene clip is used as the
        # result unit.  If the person index is missing/empty the orchestrator
        # falls back to the scene OpenCLIP index, so this is safe when the
        # person index has not been built yet.
        self.ENABLE_PERSON_ROUTING: bool = os.getenv(
            "ENABLE_PERSON_ROUTING", "true"
        ).lower() == "true"

        # ------------------------------------------------------------------
        # Privacy / access-control switches (all safe defaults)
        # ------------------------------------------------------------------
        # ACCESS_CONTROL_ENABLED=false preserves the historical open API used
        # by the bundled frontend.  Set true only behind a trusted reverse
        # proxy that injects/strips the role headers below.
        self.ACCESS_CONTROL_ENABLED: bool = os.getenv(
            "ACCESS_CONTROL_ENABLED", "false"
        ).lower() == "true"
        self.ACCESS_ROLE_HEADER: str = os.getenv("ACCESS_ROLE_HEADER", "X-Role")
        self.ACCESS_ACTOR_HEADER: str = os.getenv("ACCESS_ACTOR_HEADER", "X-Actor")
        self.ACCESS_CAMERA_HEADER: str = os.getenv(
            "ACCESS_CAMERA_HEADER", "X-Allowed-Cameras"
        )
        self.ACCESS_ZONE_HEADER: str = os.getenv(
            "ACCESS_ZONE_HEADER", "X-Allowed-Zones"
        )
        # Optional Mongo mirror of the JSONL audit trail.  Off by default so
        # offline/single-node operation never waits on an unavailable Mongo.
        self.AUDIT_MONGO_ENABLED: bool = os.getenv(
            "AUDIT_MONGO_ENABLED", "false"
        ).lower() == "true"

        # Retention is a policy switch, not an automatic startup action.
        # RETENTION_ENABLED=false means callers must opt in explicitly.
        self.RETENTION_ENABLED: bool = os.getenv(
            "RETENTION_ENABLED", "false"
        ).lower() == "true"
        self.RETENTION_HOURS: float = float(os.getenv("RETENTION_HOURS", "8"))
        self.FACE_BLUR_ENABLED: bool = os.getenv(
            "FACE_BLUR_ENABLED", "false"
        ).lower() == "true"
        # Gender is a soft attribute in the released detector.  Deployments
        # that must not expose it can set INCLUDE_GENDER_ATTRIBUTES=false:
        # captions/queries then omit it (the index itself is unchanged).
        self.INCLUDE_GENDER_ATTRIBUTES: bool = os.getenv(
            "INCLUDE_GENDER_ATTRIBUTES", "true"
        ).lower() == "true"

    def default_model_for(self, provider: str) -> str:
        """Return the model id that belongs to *provider*.

        Every provider has its own default so an opencode model id can never
        be sent to the hosted APIs:

        * ``opencode``          -> ``OPENCODE_MODEL`` (``NL_DEFAULT_MODEL``
          is the generic fallback for the opencode runtime)
        * ``openai``            -> ``gpt-4o-mini``
        * ``openrouter``        -> ``openai/gpt-4o-mini``
        * ``ollama``            -> ``llama3.1``
        * OpenAI-compatible    -> "" (the model id has to be configured)

        An empty string means "the provider decides", which lets the
        LangChain path fall back to its own per-provider default.
        """
        name = (provider or "").strip().lower()
        if name == "opencode":
            return self.OPENCODE_MODEL or self.NL_DEFAULT_MODEL
        if name == "openai":
            return os.getenv("NL_OPENAI_MODEL", "gpt-4o-mini")
        if name == "openrouter":
            return os.getenv("NL_OPENROUTER_MODEL", "openai/gpt-4o-mini")
        if name == "ollama":
            return os.getenv("NL_OLLAMA_MODEL", "llama3.1")
        return ""

    def get_active_llm_config(self) -> dict:
        """Fetch active LLM configuration from MongoDB (overrides .env if present).

        The lookup is cached so offline evaluation does not pay a Mongo
        server-selection timeout on every parsed query.  If Mongo is down we
        fall back to the environment configuration after one short attempt.
        """
        if self._llm_config_checked:
            return dict(self._llm_config_cache or {})
        self._llm_config_checked = True
        fallback = {
            "provider": self.LLM_PROVIDER,
            "model": self.default_model_for(self.LLM_PROVIDER),
            "api_key": self.OPENAI_API_KEY,
        }
        try:
            from backend.app.db.mongo import app_settings
            doc = app_settings.find_one(
                {"key": "llm_config"},
                {"_id": 0, "value": 1},
                max_time_ms=self.MONGO_SERVER_SELECTION_TIMEOUT_MS,
            )
            if doc and isinstance(doc.get("value"), dict):
                cfg = doc["value"]
                provider = cfg.get("provider") or self.LLM_PROVIDER
                # If provider is openrouter but base url is not set, it should use the default
                self._llm_config_cache = {
                    "provider": provider,
                    "model": cfg.get("model") or self.default_model_for(provider),
                    "api_key": cfg.get("api_key", self.OPENAI_API_KEY),
                }
                return dict(self._llm_config_cache)
        except Exception as e:
            # Fallback to env if DB not available or table doesn't exist yet.
            logger.warning(
                "MongoDB unavailable for llm_config ({}); using environment configuration.",
                type(e).__name__,
            )
        self._llm_config_cache = fallback
        return dict(fallback)

    def ensure_storage_dirs(self) -> None:
        self.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        self.CLIPS_DIR.mkdir(parents=True, exist_ok=True)
        self.SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        # Semantic index directories
        try:
            self.FAISS_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            self.FAISS_PERSON_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


settings = Settings()
# Create required directories on import so the backend can start writing media immediately.
settings.ensure_storage_dirs()
