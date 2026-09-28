"""Central configuration for the RAG module.

Keys come from environment variables the distillery already uses:
SILICONFLOW_API_KEY for embeddings, ZHIPU_API_KEY for generation.
Both endpoints are free-tier and reachable from CN without a proxy.
"""
import os
from pathlib import Path

# --- embeddings: SiliconFlow BAAI/bge-m3 (free tier, 1024 dims) ---
SILICONFLOW_EMBED_URL = "https://api.siliconflow.cn/v1/embeddings"
SILICONFLOW_EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", "BAAI/bge-m3")

# --- generation: Zhipu glm-4-flash (free tier, OpenAI-compatible) ---
ZHIPU_CHAT_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
ZHIPU_CHAT_MODEL = os.environ.get("RAG_CHAT_MODEL", "glm-4-flash")

# --- retrieval / chunking defaults ---
DEFAULT_DB_DIR = Path(__file__).resolve().parent.parent / "rag_data"
COLLECTION_NAME = "distillery"
TOP_K = 4
CHUNK_MAX_CHARS = 1200
CHUNK_OVERLAP_SEGMENTS = 1

# --- http behaviour ---
EMBED_BATCH = 32
HTTP_TIMEOUT = 120
CHAT_MAX_TOKENS = 1024
CHAT_TEMPERATURE = 0.2


def require_key(env: str) -> str:
    value = os.environ.get(env)
    if not value:
        raise SystemExit(f"missing environment variable {env}")
    return value
