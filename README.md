# Portfolio Assistant

A RAG-powered portfolio chat assistant with per-session short-term memory,
cross-session long-term memory, and Postgres persistence. Runs identically in
VS Code (local) and Google Colab (notebook + ngrok).

## Architecture

Built around the SOLID principles:

| Principle | How it is applied |
|---|---|
| **S - Single Responsibility** | Every class has one job: `DocumentLoaderService` loads, `DocumentChunker` splits, `AssistantPromptBuilder` builds prompts, `ContactExtractor` extracts contact info, `IdentityManager` resolves identity, `PostgresDatabase` persists. |
| **O - Open/Closed** | New providers are added by implementing the same protocols/interfaces (e.g. a new embedding provider just implements LangChain's `Embeddings`), without changing consumers. |
| **L - Liskov Substitution** | `VercelEmbeddings` is a drop-in `Embeddings`; `PortfolioAssistant` works with any `BaseChatModel` and `BaseRetriever`. |
| **I - Interface Segregation** | `core/interfaces.py` defines small protocols (`ChatRepository`) and data objects (`Turn`, `ContactInfo`, `UserProfile`); consumers depend only on what they use. |
| **D - Dependency Inversion** | `PortfolioAssistant`, `IdentityManager`, and the routes depend on abstractions. `AppContainer` (composition root) wires concrete implementations. |

## Project structure

```
portfolio-assistant/
├── config/               # Settings (env / .env / Colab userdata fallback)
├── core/                 # Logging, tracing, shared protocols + data objects
├── embeddings/           # VercelEmbeddings (Embeddings implementation)
├── llm/                  # NovitaChatModel
├── vectorstore/          # PineconeVectorStoreProvider (index lifecycle + store)
├── documents/            # Loaders, chunker, ingestion pipeline
├── rag/                  # Prompt builder + PortfolioAssistant
├── memory/               # Short-term memory, contact extraction, identity
├── persistence/          # PostgresDatabase (implements ChatRepository)
├── api/                  # Schemas, container (composition root), routes, app factory
│   └── routes/           # chat.py, session.py, admin.py, ui.py
├── colab/                # ServerManager (background uvicorn + ngrok for Colab)
├── scripts/              # CLI ingestion
├── static/               # Chat UI (single-page HTML)
├── tests/                # Unit tests
├── notebooks/            # Colab runner notebook
├── main.py               # Entry point
└── requirements.txt
```

## Setup (VS Code / local)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                 # fill in your keys
python -m scripts.ingest_document --file data/resume.pdf --type pdf
uvicorn main:app --reload --port 8000
```

Open http://127.0.0.1:8000 for the chat UI, or http://127.0.0.1:8000/docs for Swagger.

Run tests:

```bash
python -m pytest tests/
```

## Setup (Google Colab)

1. Upload the project folder to Colab (or clone your repo).
2. Open `notebooks/portfolio_assistant_colab.ipynb` and run the cells in order.
   Secrets are read from Colab userdata with the same names as the `.env` variables,
   so no code changes are needed.

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Chat UI |
| GET | `/health` | Health check |
| POST | `/chat` | Chat (single response) |
| POST | `/chat/stream` | Chat (streaming) |
| POST | `/session/start` | Start/resume session (identity form) |
| DELETE | `/session/{id}` | Clear session (flushes long-term memory first) |
| GET | `/contact/{session_id}` | Contact info for a session (owner view) |
| GET | `/profiles` | All known users + long-term memory (owner view) |
| GET | `/history/{session_id}` | Session message log (owner view) |

## Environment variables

See `.env.example`. The same names are used as Colab userdata secret names.
