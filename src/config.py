"""
Central configuration for DevBrain.

Every module imports settings from here instead of reading .env directly.
This gives us one place to validate, set defaults, and rename variables.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ── Load .env file ────────────────────────────────────────────────
# find_dotenv walks up directories to locate .env, but we'll be explicit:
# .env lives at the project root (same level as pyproject.toml)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


# ── Helper ────────────────────────────────────────────────────────
def _require_env(name: str) -> str:
    """
    Get a required environment variable or exit with a clear error.

    Why exit instead of raise?
    Because a missing API key means nothing will work — we want
    a loud, immediate failure with instructions, not a traceback
    20 lines deep that says 'NoneType has no attribute split'.
    """
    value = os.getenv(name)
    if not value:
        print(f"\n❌ Missing required environment variable: {name}")
        print(f"   → Copy .env.example to .env and fill in your keys:")
        print(f"     cp .env.example .env\n")
        sys.exit(1)
    return value


# ── API Keys ──────────────────────────────────────────────────────
OPENAI_API_KEY: str = _require_env("OPENAI_API_KEY")
PINECONE_API_KEY: str = _require_env("PINECONE_API_KEY")

# GitHub token is optional but strongly recommended (60 req/hr without, 5000 with)
GITHUB_TOKEN: str | None = os.getenv("GITHUB_TOKEN")


# ── Pinecone ──────────────────────────────────────────────────────
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "devbrain")

# Embedding dimension — must match the embedding model's output
# text-embedding-3-small = 1536 dimensions
EMBEDDING_DIMENSION: int = 1536

# Similarity metric — cosine is standard for text embeddings
PINECONE_METRIC: str = "cosine"

# Namespaces — one per source type, keeps data organized and searchable
NAMESPACE_DOCS: str = "docs"
NAMESPACE_ISSUES: str = "issues"


# ── OpenAI ────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Batch size for embedding API calls
# OpenAI allows up to 2048 texts per call, but 100 is safer for rate limits
EMBEDDING_BATCH_SIZE: int = 100


# ── GitHub (target repo) ─────────────────────────────────────────
TARGET_REPO_OWNER: str = os.getenv("TARGET_REPO_OWNER", "fastapi")
TARGET_REPO_NAME: str = os.getenv("TARGET_REPO_NAME", "fastapi")

# Docs path inside the repo (where markdown files live)
TARGET_DOCS_PATH: str = "docs/en/docs"

# How many issues to fetch (most recent, sorted by updated_at)
MAX_ISSUES: int = 500


# ── Chunking ──────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))


# ── Data directories ──────────────────────────────────────────────
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DOCS_DIR: Path = DATA_DIR / "raw" / "docs"
RAW_ISSUES_DIR: Path = DATA_DIR / "raw" / "issues"

# Create directories if they don't exist
RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
RAW_ISSUES_DIR.mkdir(parents=True, exist_ok=True)


# ── Quick sanity check ────────────────────────────────────────────
if __name__ == "__main__":
    """Run this file directly to verify your config is loaded correctly."""
    from rich import print as rprint

    rprint("[bold green]✅ Configuration loaded successfully![/bold green]\n")
    rprint(f"  OpenAI API Key:    {OPENAI_API_KEY[:8]}...{OPENAI_API_KEY[-4:]}")
    rprint(f"  Pinecone API Key:  {PINECONE_API_KEY[:8]}...{PINECONE_API_KEY[-4:]}")
    rprint(f"  GitHub Token:      {'✅ Set' if GITHUB_TOKEN else '⚠️  Not set (60 req/hr limit)'}")
    rprint(f"  Pinecone Index:    {PINECONE_INDEX_NAME}")
    rprint(f"  Embedding Model:   {EMBEDDING_MODEL}")
    rprint(f"  Target Repo:       {TARGET_REPO_OWNER}/{TARGET_REPO_NAME}")
    rprint(f"  Chunk Size:        {CHUNK_SIZE} chars, {CHUNK_OVERLAP} overlap")
    rprint(f"  Data Dir:          {DATA_DIR}")