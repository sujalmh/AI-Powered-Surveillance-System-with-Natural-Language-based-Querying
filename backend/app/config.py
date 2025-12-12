import os
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
        self.LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "")  # e.g., "openai", "ollama"
        self.OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        self.OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.NL_DEFAULT_MODEL: str = os.getenv("NL_DEFAULT_MODEL", "gpt-4o-mini")

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
