# Course Notes — arXiv Paper Curator (Production Agentic RAG)

Study notes for each week of the course. Week 1 is covered by the original blog post; Weeks 2–7 are written from the code in this repo (each week's git tag, plus `main`).

> The repo has **7 weeks** of content (tags `week1.0` … `week7.0`), even though the original blog series said "Week 1 of 6". Week 6 (monitoring + caching) and Week 7 (agentic RAG + Telegram) were both added later.

| Week | Theme | Notes | Code to study | Notebook |
|------|-------|-------|---------------|----------|
| 1 | Infrastructure foundation | [Blog post](https://jamwithai.substack.com/p/the-infrastructure-that-powers-rag) | `git checkout week1.0` | `notebooks/week1/week1_setup.ipynb` |
| 2 | Data ingestion pipeline | [week2-data-ingestion.md](week2-data-ingestion.md) | `git checkout week2.0` | `notebooks/week2/week2_arxiv_integration.ipynb` |
| 3 | BM25 keyword search | [week3-keyword-search.md](week3-keyword-search.md) | `git checkout week3.0` | `notebooks/week3/week3_opensearch.ipynb` |
| 4 | Chunking + hybrid search | [week4-chunking-hybrid-search.md](week4-chunking-hybrid-search.md) | `git checkout week4.0` | `notebooks/week4/week4_hybrid_search.ipynb` |
| 5 | Complete RAG with a local LLM | [week5-complete-rag.md](week5-complete-rag.md) | `git checkout week5.0` | `notebooks/week5/week5_complete_rag_system.ipynb` |
| 6 | Monitoring (Langfuse) + caching (Redis) | [week6-monitoring-caching.md](week6-monitoring-caching.md) | `git checkout week6.0` | `notebooks/week6/week6_cache_testing.ipynb` |
| 7 | Agentic RAG (LangGraph) + Telegram bot | [week7-agentic-rag-telegram.md](week7-agentic-rag-telegram.md) | `git checkout week7.0` | `notebooks/week7/week7_agentic_rag.ipynb` |

## How the system grows week by week

```
Week 1  Docker Compose: FastAPI · PostgreSQL · OpenSearch · Airflow · Ollama
          │
Week 2  arXiv API ──► PDF download ──► Docling parse ──► PostgreSQL     (Airflow, daily)
          │
Week 3  PostgreSQL ──► OpenSearch (paper-level index) ──► BM25 /search
          │
Week 4  sections ──► chunks ──► Jina embeddings ──► OpenSearch (chunk index, kNN + BM25)
                                                  └► /hybrid-search  (RRF fusion)
          │
Week 5  query ──► hybrid retrieval ──► prompt ──► Ollama ──► /ask, /stream, Gradio UI
          │
Week 6  + Redis exact-match cache in front, + Langfuse traces around every step
          │
Week 7  + LangGraph agent: guardrail → retrieve → grade → (rewrite → retrieve) → answer
        + Telegram bot as a second front-end
```

## How to use these notes

1. Read the week's notes once, end to end.
2. Check out that week's tag (`git checkout week3.0`) and read the files listed under **Code map**. The notes describe `main`; where a week's tag differs from `main`, the notes say so.
3. Run that week's notebook.
4. Answer the **Self-check** questions without looking.
5. Read **Gotchas**. Several of them are real bugs on `main`, and you learn a lot by fixing them.

## Bugs on `main` you should know about before running anything

These come from reading the code (not from running it). Details are in the linked notes.

| Severity | Issue | Where | Week |
|----------|-------|-------|------|
| 🔴 High | PDF parsing always fails: `parse_pdf` became sync in commit `1a7ab20`, but its caller still `await`s it, so papers get stored without text and nothing gets chunked | `src/services/pdf_parser/parser.py:33` | [2](week2-data-ingestion.md#gotchas) |
| 🔴 High | `/api/v1/ask` and `/api/v1/stream` crash: `RAGTracer` calls `trace_rag_request` and `create_span`, which were removed from `LangfuseTracer` in the Week 7 commit | `src/services/langfuse/tracer.py:21` | [6](week6-monitoring-caching.md#gotchas) |
| 🔴 High | Agentic RAG answers every question with "out of scope": `OllamaClient.get_langchain_model` is never defined, so the guardrail falls back to score 50, which is below the threshold of 60 | `src/services/agents/nodes/guardrail_node.py:84` | [7](week7-agentic-rag-telegram.md#gotchas) |
| 🟠 Medium | Langfuse keys in `.env.example` use `LANGFUSE_` but the settings class reads `LANGFUSE__` (double underscore), so tracing stays off silently | `.env.example:61-67` vs `src/config.py:108` | [6](week6-monitoring-caching.md#gotchas) |
| 🟡 Low | Text shorter than 100 words crashes the chunker (`_reconstruct_text` gets 2 args but takes 1) | `src/services/indexing/text_chunker.py:114` | [4](week4-chunking-hybrid-search.md#gotchas) |

To study each week as it was released (before later weeks changed things), use the week's tag.
