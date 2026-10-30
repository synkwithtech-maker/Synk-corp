# Synk Corp Customer Support API & AI Assistant

An enterprise-ready customer support chatbot built with **FastAPI**, **LangGraph**, **Qdrant Vector Database (RAG)**, **Defense-in-Depth Guardrails**, and **Redis Conversation Memory**.

---

## 🚀 Quickstart & Setup Guide

Follow these steps to set up and run the project locally.

### 1. Prerequisites
* **Python**: Python 3.10+ (tested on Python 3.10 – 3.14)
* **Docker** (recommended for running Redis) or a local Redis server
* **LLM API Key**: [OpenRouter API Key](https://openrouter.ai) or OpenAI API Key

---

### 2. Set Up Virtual Environment & Dependencies

```bash
# Clone the repository
git clone https://github.com/synkwithtech-maker/Synk-corp.git
cd Synk-corp

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS / Linux:
source venv/bin/activate

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# On Windows (Command Prompt):
.\venv\Scripts\activate.bat

# Install required dependencies
pip install -r requirements.txt
```

---

### 3. Configure Environment Variables

Copy `.env.example` to create your local `.env` file:

```bash
# On Linux / macOS:
cp .env.example .env

# On Windows (PowerShell / CMD):
copy .env.example .env
```

Open `.env` and fill in your settings:
* `OPENROUTER_API_KEY`: Your OpenRouter API key (or `OPENAI_API_KEY`).
* `OPENROUTER_MODEL`: LLM model identifier (e.g. `qwen/qwen3.8-27b`).
* `EMBEDDING_MODEL_NAME`: Embedding model (e.g. `sentence-transformers/all-MiniLM-L6-v2`).
* Adjust Redis or Qdrant parameters if using custom hostnames or ports.

---

### 4. Start Redis

Start a Redis container via Docker:

```bash
docker run -d --name synk-redis -p 6379:6379 redis:alpine
```
*(Or run in foreground: `docker run -p 6379:6379 redis`)*

> **Note:** If Redis is offline, the service automatically falls back to an in-memory session manager for local development without failing.

---

### 5. Ingest Knowledge Base Documents

Place your reference documents (`.txt` or `.md`) into the `documents/` directory:

```bash
# Create directory if it does not exist
mkdir documents

# Place your documents inside (e.g. documents/synk_company_info.txt)
```

Run the ingestion script to generate embeddings and populate the local Qdrant vector database:

```bash
python ingest.py
```
*Expected output:* `[OK] Ingestion complete. <N> chunks indexed.`

---

### 6. Start the API Server

Launch the FastAPI service using Uvicorn:

```bash
uvicorn api:app --reload
```

Once started, access the service:
* **API Root:** `http://localhost:8000`
* **Interactive Swagger UI:** `http://localhost:8000/docs`
* **Alternative API Docs (ReDoc):** `http://localhost:8000/redoc`
* **Health Probes:** `http://localhost:8000/health`

---

## 🌟 Key Architectural Features

1. **Defense-in-Depth Guardrails (`guardrails.py`)**:
   - **Prompt Injection & Jailbreak Prevention**: Blocks instruction overrides (`ignore previous instructions`, `DAN`, persona hijacks) before they reach the vector database or LLM.
   - **PII Redaction**: Automatically detects and masks sensitive data (Credit Cards, SSNs, API secrets) to protect customer privacy and prevent memory leaks.
   - **System Prompt Leak Suppression**: Inspects generated output to ensure internal prompt instructions and model delimiters are never regurgitated.
   - **Strict Grounding Enforcement**: Guarantees answers strictly reflect Synk Corp documents, falling back to a safe refusal when information is absent.

2. **Redis Conversation Memory (`memory.py`)**:
   - Multi-turn conversational memory partitioned by `session_id`.
   - **Sliding TTL Expiration**: Session keys automatically expire (default: 24h) and refresh on user activity.
   - **Sliding History Window**: Injects recent dialogue turns within context budget.
   - **Graceful Fallback**: If Redis is offline or undergoing maintenance, seamlessly falls back to an in-memory session manager without dropping requests or crashing the server.

3. **RAG & Vector Search (`rag.py`)**:
   - Decoupled, lazily initialized vector store preventing lock contention and startup freezes.
   - Grounded context retrieval from Synk Corp corporate documents stored in Qdrant.
   - Source document attribution and hallucination guardrails.

4. **Guarded LangGraph Workflow (`chatbot.py`)**:
   - StateGraph architecture with conditional routing:
     - `START` -> `input_guardrail` -> `[if safe: retrieve -> answer -> output_guardrail -> END] | [if blocked: guardrail_block -> END]`
   - Multi-lingual support: English, Nepali, and Romanized Nepali.

5. **Production FastAPI Service (`api.py`)**:
   - Async thread execution for CPU/network-bound LLM and RAG tasks.
   - Modern `lifespan` application lifecycle.
   - CORS middleware configured for frontend web and mobile clients.
   - OpenAPI / Swagger documentation (`/docs` and `/redoc`).
   - Detailed readiness & liveness probes (`/health`) including memory, RAG, and guardrails statuses.

---

## 🛡️ Guardrails Architecture

```mermaid
flowchart TD
    User([Customer]) -->|Query| InputGuard[Input Guardrail]
    InputGuard -->|Mask PII| CleanQuery[Sanitized Query]
    InputGuard -->|Injection / Jailbreak Detected| BlockNode[Fast Block Node]
    BlockNode -->|Safe Policy Refusal| User

    CleanQuery -->|Fetch Context| Retriever[Qdrant Retriever]
    CleanQuery -->|Fetch Memory| Redis[(Redis Memory)]
    Retriever --> Generator[LLM Generator]
    Redis --> Generator
    Generator --> OutputGuard[Output Guardrail]
    OutputGuard -->|Leak Check & Grounding Check| FinalAnswer[Safe Answer]
    FinalAnswer -->|ChatResponse| User
```

---

## 📡 API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Operational status and API metadata. |
| `GET` | `/health` | Health probe (API, Redis connection status & latency, Qdrant readiness, Guardrails info). |
| `POST` | `/chat` | Submit question with `session_id`; returns answer, referenced sources, and guardrail classification. |
| `GET` | `/chat/{session_id}/history` | Retrieve complete dialogue history stored in Redis for that session. |
| `DELETE` | `/chat/{session_id}` | Clear conversation history from Redis. |
| `GET` | `/docs` | Interactive Swagger UI documentation. |

### Example Chat Request:

```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "session_id": "client-123",
       "question": "What services does Synk Corp provide?"
     }'
```

### Example Chat Response:

```json
{
  "session_id": "client-123",
  "question": "What services does Synk Corp provide?",
  "answer": "Synk Corp specializes in AI consulting, custom software development...",
  "sources": ["synk_company_info.txt"],
  "guardrail_passed": true,
  "guardrail_category": "clean",
  "latency_ms": 1150.4
}
```
