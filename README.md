
<div align="center">
  <h1>AgenticRAG 🧠</h1>
  <p><strong>Local-first RAG Agent with Dual Retrieval + ReAct Reasoning + Tiered Memory</strong></p>
</div>

<p align="center">
  <a href="./README.md">English</a> |
  <a href="./README_CN.md">简体中文</a>
</p>

<p align="center">
  <a href="https://github.com/hyx1249207016-netizen/AgenticRAG/actions/workflows/ci.yml">
    <img src="https://github.com/hyx1249207016-netizen/AgenticRAG/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
  </a>
  <a href="https://python.org">
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python" alt="Python">
  </a>
  <a href="https://ollama.com">
    <img src="https://img.shields.io/badge/Ollama-Qwen3.5-5B5?logo=ollama" alt="Ollama">
  </a>
</p>

---

**AgenticRAG** is a local-first, production-ready AI agent that combines **dual-channel retrieval** (semantic + BM25), **ReAct reasoning**, and **tiered memory** (short-term, long-term, and user profiling). It runs entirely on your local machine with Ollama-no cloud API, no data leakage, no extra cost.

Designed as a general-purpose knowledge Q&A agent, it ships with a campus-scenario demo, but you can plug in **any knowledge base** (company docs, personal notes, research papers, product manuals) and get an intelligent assistant in minutes.

---

## ✨ Features

| Feature | Description |
|---|---|
| **ReAct Agent** | LangGraph-powered tool-use loop with 4 built-in tools: knowledge base search, web search, web fetch, and clock |
| **Dual-Channel Retrieval** | Semantic vector search (ChromaDB + BGE embeddings) + BM25 keyword search, fused by bge-reranker |
| **Tiered Memory** | Short-term (recent 10 rounds) + ChromaDB long-term summaries + user profiling-never loses context |
| **SSRF Protection** | Dual IP validation (Python + nslookup) prevents server-side request forgery attacks |
| **Graceful Degradation** | Automatic fallback: vector BM25 direct LLM, with circuit breaker and retry |
| **Streaming Output** | SSE-powered real-time streaming via FastAPI |
| **Provider-Agnostic** | Switch models via config-Ollama, OpenAI-compatible, or any LangChain-supported LLM |
| **Docker Support** | One-command deployment with docker-compose |
| **Evaluation Suite** | Built-in RAG accuracy evaluation script for iterative improvement |

---

## Quick Start

### Prerequisites

- [Ollama](https://ollama.com) installed and running
- Python 3.10+

### 1. Clone and Setup

```bash
git clone https://github.com/hyx1249207016-netizen/AgenticRAG.git
cd AgenticRAG

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Pull Models

```bash
ollama pull qwen3.5:0.5b
ollama pull bge-m3:latest
```

Create the custom model profile:

```bash
ollama create agenticrag-model -f Modelfile.qwen35
```

### 3. Configure

```bash
cp .env.example .env
```

### 4. Load Knowledge Base

Place your markdown documents in the `data/` directory, or use the built-in sample data.

### 5. Run

```bash
python main.py
```

Open your browser at [http://localhost:8000](http://localhost:8000)

> Docker users: Run `docker compose up -d` for containerized deployment.

---

## Configuration

Key environment variables in `.env`:

| Variable | Default | Description |
|---|---|---|
| OLLAMA_BASE_URL | http://localhost:11434 | Ollama service URL |
| CHAT_MODEL | agenticrag-model | LLM for conversation |
| EMBEDDING_MODEL | bge-m3:latest | Embedding model |
| MEMORY_PERSIST_DIR | ./data/memory_db | Long-term memory storage |

See [.env.example](.env.example) for all options.

---

## Evaluation

```bash
python eval.py
```

Runs RAG accuracy evaluation across built-in test cases and outputs precision/recall metrics.

---

## Project Structure

```
AgenticRAG/
├── main.py                     # FastAPI entry point
├── app/
│   ├── api/routes.py           # Chat and file upload endpoints
│   ├── core/config.py          # Environment configuration
│   ├── services/
│   │   ├── memory_service.py   # Tiered memory system
│   │   ├── react_agent.py      # LangGraph ReAct agent
│   │   ├── reranker.py         # bge-reranker fusion
│   │   └── vector_store.py     # ChromaDB management
│   └── tools/agent_tools.py    # Tool implementations
├── data/                       # Knowledge base documents
├── static/                     # Web frontend
├── tests/                      # Test suite
└── docker-compose.yml          # Docker deployment
```

---

## Security

- **SSRF Protection**: Every outbound request undergoes dual IP validation-Python urllib + system nslookup-blocking private/internal IP ranges before any connection is made.
- **Input Validation**: File uploads are type-checked; API inputs are validated via Pydantic schemas.
- **No Telemetry**: Zero data collection. Everything runs locally.

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

---

## License

[MIT](LICENSE) 2025 AgenticRAG

---

<p align="center">
  <sub>Built with LangGraph, ChromaDB, FastAPI, and Ollama</sub>
</p>
