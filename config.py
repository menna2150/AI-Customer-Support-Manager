import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent

# --- Provider selection --------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")


def _detect_provider() -> str:
    explicit = os.getenv("LLM_PROVIDER", "").strip().lower()
    if explicit in {"groq", "anthropic"}:
        return explicit
    if GROQ_API_KEY:
        return "groq"
    if ANTHROPIC_API_KEY:
        return "anthropic"
    return "groq"  # default; will error on first call if no key


LLM_PROVIDER = _detect_provider()

# --- Vector store --------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DIR = os.getenv("CHROMA_DIR", str(ROOT / "knowledge_base" / "store"))
KB_COLLECTION = os.getenv("KB_COLLECTION", "support_kb")

KB_DOCS_DIR = ROOT / "knowledge_base" / "docs"

# --- Pipeline tunables ---------------------------------------------------
QA_MAX_RETRIES = 2
RAG_TOP_K = 4
LLM_RETRIES = 2  # how many times to retry structured() on validation failure
