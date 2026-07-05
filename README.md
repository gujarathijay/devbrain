# DevBrain — Multi-Source Developer Support Intelligence Platform

An Advanced RAG application that ingests a project's **official docs**, **GitHub Issues**, and **Discussions** to answer developer questions with precise citations.

**Demo dataset:** [FastAPI](https://github.com/fastapi/fastapi) — Python web framework.

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangChain + LangGraph |
| Vector DB | Pinecone Serverless |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI GPT-4o-mini / GPT-4o |
| Reranking | Cohere Rerank v3 |
| Evaluation | Braintrust |
| Backend | FastAPI |
| Frontend | Next.js + TypeScript + Tailwind |
| Deployment | Docker + AWS |

## Setup

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- API keys: OpenAI, Pinecone, GitHub token

### Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/devbrain.git
cd devbrain

# Create virtual environment and install dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run the ingestion pipeline (Phase 1)
uv run python scripts/run_ingestion.py

# Verify data is in Pinecone
uv run python scripts/verify_pinecone.py
```

## Project Status

- [x] Phase 1: Data ingestion pipeline + vector storage
- [ ] Phase 2: Basic retrieval chain + CLI Q&A
- [ ] Phase 3: Advanced retrieval (reranking, hybrid search)
- [ ] Phase 4: Agentic RAG with LangGraph
- [ ] Phase 5: Evaluation pipeline (Braintrust)
- [ ] Phase 6: FastAPI backend
- [ ] Phase 7: Next.js frontend
- [ ] Phase 8: Docker + AWS deployment + CI/CD