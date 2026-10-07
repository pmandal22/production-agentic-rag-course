# Production Agentic RAG — My Course Workthrough

This is my personal working copy of the **[Production Agentic RAG course](https://github.com/jamwithai/production-agentic-rag-course)** (the "arXiv Paper Curator", Phase 1 of *The Mother of AI Project*) by **[Shirin Khosravi Jam](https://www.linkedin.com/in/shirin-khosravi-jam/)** and **[Shantanu Ladhwe](https://www.linkedin.com/in/shantanuladhwe/)** of [Jam With AI](https://jamwithai.substack.com/).

**I did not create this course.** The architecture, code and learning material are theirs. This repo holds my run-through: my study notes, the notebooks as I ran them, and the fixes I needed to get things working locally.

> For the official course, its weekly releases, and updates, use the original repository:
> **https://github.com/jamwithai/production-agentic-rag-course**

<p align="center">
  <img src="static/mother_of_ai_project_rag_architecture.gif" alt="RAG Architecture (from the original course)" width="700">
</p>

---

## What the course builds

A research assistant that fetches arXiv papers, indexes them, and answers questions about them with RAG, built up over seven weeks:

| Week | Topic | Original blog post |
|------|-------|--------------------|
| 1 | Infrastructure: Docker Compose, FastAPI, PostgreSQL, OpenSearch, Airflow, Ollama | [The Infrastructure That Powers RAG Systems](https://jamwithai.substack.com/p/the-infrastructure-that-powers-rag) |
| 2 | Data ingestion: arXiv API, Docling PDF parsing, Airflow DAGs | [Building Data Ingestion Pipelines for RAG](https://jamwithai.substack.com/p/bringing-your-rag-system-to-life) |
| 3 | BM25 keyword search with OpenSearch | [The Search Foundation Every RAG System Needs](https://jamwithai.substack.com/p/the-search-foundation-every-rag-system) |
| 4 | Chunking + hybrid search (Jina embeddings, RRF) | [The Chunking Strategy That Makes Hybrid Search Work](https://jamwithai.substack.com/p/chunking-strategies-and-hybrid-rag) |
| 5 | Complete RAG with a local LLM, streaming, Gradio UI | [The Complete RAG System](https://jamwithai.substack.com/p/the-complete-rag-system) |
| 6 | Monitoring (Langfuse) + caching (Redis) | [Production-ready RAG: Monitoring & Caching](https://jamwithai.substack.com/p/production-ready-rag-monitoring-and) |
| 7 | Agentic RAG with LangGraph + Telegram bot | [Agentic RAG with LangGraph and Telegram](https://jamwithai.substack.com/p/agentic-rag-with-langgraph-and-telegram) |

The original README has the full weekly walkthroughs, architecture diagrams and API reference.

---

## My notes

Study notes for each week are in [`notes/`](notes/README.md):

- [Week 2 — Data ingestion](notes/week2-data-ingestion.md)
- [Week 3 — Keyword search](notes/week3-keyword-search.md)
- [Week 4 — Chunking & hybrid search](notes/week4-chunking-hybrid-search.md)
- [Week 5 — Complete RAG](notes/week5-complete-rag.md)
- [Week 6 — Monitoring & caching](notes/week6-monitoring-caching.md)
- [Week 7 — Agentic RAG & Telegram](notes/week7-agentic-rag-telegram.md)

The notebooks under [`notebooks/`](notebooks/) include my run outputs.

---

## Changes from the original

Fixes I made while working through the course:

**Week 7 agentic endpoint (`/api/v1/ask-agentic` returned 500)**
- `src/services/agents/factory.py`: `make_agentic_rag_service()` now accepts the `model` argument that `src/dependencies.py` passes to it.
- `src/services/langfuse/client.py`: added `create_span`, `end_span` and `trace_rag_request`, which the agent nodes and RAG tracer call but were missing.
- `src/services/ollama/client.py`: added `get_langchain_model()` (returns a `ChatOllama`), which the guardrail, grading, rewrite and answer nodes call. Without it, the guardrail silently fell back to a fixed score of 50.

**Configuration**
- `.env.example` / `compose.yml`: Langfuse settings use the double-underscore `LANGFUSE__*` names that `src/config.py` reads, and the local Langfuse project is pre-created with matching dev keys (`pk-lf-local-dev` / `sk-lf-local-dev`).

**PDF parsing**
- `src/services/pdf_parser/parser.py`: Docling parsing runs in a worker thread so it doesn't block the event loop.

**Known issue:** inside the `api` container the Langfuse SDK still reads `LANGFUSE_BASE_URL=http://localhost:3001` from `.env`, so traces aren't sent. Adding `LANGFUSE_BASE_URL=http://langfuse-web:3000` to the `api` service environment in `compose.yml` should fix it.

---

## Running it locally

Prerequisites: Docker Desktop, Python 3.12+, [uv](https://docs.astral.sh/uv/getting-started/installation/), 8 GB+ RAM, 20 GB+ free disk.

```bash
git clone https://github.com/pmandal22/production-agentic-rag-course.git
cd production-agentic-rag-course

cp .env.example .env      # add your JINA_API_KEY (Week 4+) and TELEGRAM__BOT_TOKEN (Week 7)
uv sync
docker compose up --build -d

curl http://localhost:8000/api/v1/health
```

The `api` image copies `src/` in at build time, so rebuild after code changes: `docker compose up -d --build api`.

| Service | URL |
|---------|-----|
| API docs | http://localhost:8000/docs |
| Gradio UI (`uv run python gradio_launcher.py`) | http://localhost:7861 |
| Langfuse | http://localhost:3001 |
| Airflow | http://localhost:8080 |
| OpenSearch Dashboards | http://localhost:5601 |

The week tags (`week1.0` … `week7.0`) mentioned in my notes are in the [original repository](https://github.com/jamwithai/production-agentic-rag-course/tags), not in this one.

---

## Credits & license

Course, code and diagrams © 2025 [Jam With AI](https://jamwithai.substack.com/), created by Shirin Khosravi Jam and Shantanu Ladhwe, released under the MIT License (see [LICENSE](LICENSE)). My notes and fixes in this repo are released under the same license.
