# AutoClaim AI — 100% Local Open-Source Multi-Agent Car Damage & Insurance RAG

A runnable, 100% local, open-source starter application for **car damage severity assessment and insurance claim triage**. It combines a **LangGraph** multi-agent workflow, **Ollama** local vision/text models, **Qdrant** vector RAG, **AI governance & evaluation**, a **FastAPI** microservice backend, a **ReactJS** frontend, and **Docker Compose + GitHub Actions CI**.

## Architecture

1. **Guardrail & Intake Agent** (`backend/app/services/governance.py`): Scans claim descriptions for PII (emails, phone numbers, SSNs, license plates) and redacts them before LLM processing.
2. **Vision Damage Assessor Agent** (`backend/app/services/ollama_client.py`): Uses local [Ollama `qwen2.5vl:7b`](https://ollama.com/library/qwen2.5vl) (or [`llama3.2-vision`](https://ollama.com/library/llama3.2-vision)) to inspect the uploaded car photo and extract damaged parts, damage types, structural risk, and confidence.
3. **Policy & Repair RAG Agent** (`backend/app/services/rag_service.py`): Embeds the damage summary using `nomic-embed-text` and queries [Qdrant](https://qdrant.tech/documentation/tutorials-build-essentials/agentic-rag-langgraph/) for matching policy rules, deductibles, and severity rubrics (`backend/data/policies.json`).
4. **Severity Adjudicator & Eval Judge** (`backend/app/agents/graph.py`): Uses `qwen2.5:7b` to classify severity (`MINOR`, `MODERATE`, `SEVERE`, `TOTAL_LOSS`), estimate repair ranges, compute a RAG faithfulness score, and flag human adjuster escalation when needed.

## Quick Start (Docker Compose)

```bash
cp .env.example .env
docker compose up --build -d

# Pull open-source models into the local Ollama container
docker exec -it autoclaim-ollama ollama pull qwen2.5vl:7b
docker exec -it autoclaim-ollama ollama pull qwen2.5:7b
docker exec -it autoclaim-ollama ollama pull nomic-embed-text
```

- **Frontend UI:** `http://localhost:3000`
- **FastAPI Docs:** `http://localhost:8000/docs`
- **Qdrant Dashboard:** `http://localhost:6333/dashboard`

## Local Development (Without Docker)

```bash
# 1. Pull models locally with Ollama
make pull-models

# 2. Run FastAPI backend (uses in-memory/local Qdrant or Docker Qdrant + SQLite)
cd backend
pip install -e ".[dev]"
QDRANT_IN_MEMORY=true uvicorn app.main:app --reload --port 8000

# 3. Run React frontend
cd frontend
npm install && npm run dev
```

## Running Tests & Evaluations

```bash
cd backend
pytest -v
```

You can also trigger the golden evaluation suite from the UI (**Run Local Evals**) or via `POST /api/v1/evals/run`.
