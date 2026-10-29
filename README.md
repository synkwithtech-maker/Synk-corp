# Synk Corp Customer Support API & AI Assistant

An enterprise-ready customer support chatbot built with **FastAPI**, **LangGraph**, **Qdrant Vector Database (RAG)**, **Defense-in-Depth Guardrails**, **Real-Time Token Streaming (SSE)**, **Complete RAG File Lifecycle (Upload, Update, Delete, Search)**, and **Redis Conversation Memory**.

---

## 🌟 Key Architectural Features

1. **Complete RAG Document Lifecycle APIs (`/rag/...`)**:
   - **Upload (`POST /rag/upload`)**: Upload and vectorize files (`.pdf`, `.docx`, `.csv`, `.json`, `.txt`, `.md`).
   - **Update / Replace (`PUT /rag/documents/{filename}`)**: Overwrite an existing document on disk and replace its vector points in Qdrant with updated chunks.
   - **Delete (`DELETE /rag/documents/{filename}`)**: Permanently remove a company document from both storage and the Qdrant vector database.
   - **List (`GET /rag/documents`)**: View all indexed files, sizes in KB, and modification timestamps.
   - **Test Search (`POST /rag/test-retrieval`)**: Verify vector search retrieval directly without invoking the LLM.
   - **Batch Upload (`POST /rag/upload-batch`)**: Upload and vectorize multiple documents in a single request.

2. **Real-Time Token Streaming (`/chat/stream`)**:
   - **Server-Sent Events (SSE)**: Streams tokens chunk-by-chunk in real time, reducing Time-To-First-Token (TTFT) to $< 300\text{ ms}$.
   - Available via both `POST /chat/stream` (standard frontend payload) and `GET /chat/stream` (native browser `EventSource` and URL bar testing).
   - Real-time event lifecycle: `start` $\to$ `sources` $\to$ `token` (repeated) $\to$ `done` $\to$ `[DONE]`.

3. **Defense-in-Depth AI Guardrails (`guardrails.py`)**:
   - **Prompt Injection & Jailbreak Prevention**: Blocks instruction overrides (`ignore previous instructions`, `DAN`, persona hijacks) before hitting the vector database or LLM.
   - **PII Redaction**: Automatically detects and masks sensitive data (Credit Cards, SSNs, API secrets) to protect customer privacy and prevent memory leaks.
   - **System Prompt Leak Suppression**: Inspects generated output to ensure internal prompt instructions and model delimiters are never regurgitated.
   - **Strict Grounding Enforcement**: Guarantees answers strictly reflect Synk Corp documents, falling back to a safe refusal when information is absent.

4. **Enterprise Security, Authentication & Hardening (`security.py`)**:
   - **Role-Based API Key Authentication**: Destructive RAG endpoints (`POST /rag/upload`, `PUT /rag/documents/...`, `DELETE /rag/documents/...`, `/admin/ingest/url`) require `ADMIN_API_KEY`.
   - **Client API Key (Optional)**: Chat endpoints support optional API key verification via `REQUIRE_API_KEY_FOR_CHAT`.
   - **Sliding-Window Rate Limiting (Anti-DDoS & LLM Wallet Drain Defense)**: Limits requests per client IP (30 req/min for chat, 10 req/min for admin RAG) using Redis or in-memory sliding bucket.
   - **Server-Side Request Forgery (SSRF) Protection**: Blocks malicious internal, private, loopback, and cloud metadata requests (`169.254.169.254`) on URL scraping.
   - **Path Traversal & Filename Sanitization**: Replaces dangerous traversal patterns (`../`), disallows hidden files and Windows reserved device names (`CON`, `PRN`).
   - **Upload Payload Guard**: Enforces maximum upload file size (`MAX_UPLOAD_SIZE_MB`, default 15MB) preventing memory exhaustion DoS.
   - **Session ID Injection Guard**: Enforces strict alphanumeric / dash validation on `session_id`.
   - **OWASP HTTP Security Headers**: Injects `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection`, and `Referrer-Policy`.

5. **Redis Conversation Memory (`memory.py`)**:
   - Multi-turn conversational memory partitioned by `session_id`.
   - **Sliding TTL Expiration**: Session keys automatically expire (default: 24h) and refresh on user activity.
   - **Sliding History Window**: Injects recent dialogue turns within context budget.
   - **Graceful Fallback**: If Redis is offline or undergoing maintenance, seamlessly falls back to an in-memory session manager without dropping requests or crashing the server.

5. **RAG & Vector Search (`rag.py`)**:
   - Decoupled, lazily initialized vector store preventing lock contention and startup freezes.
   - Grounded context retrieval from Synk Corp corporate documents stored in Qdrant.
   - Source document attribution and hallucination guardrails.

6. **Guarded LangGraph Workflow (`chatbot.py`)**:
   - StateGraph architecture with conditional routing:
     - `START` $\to$ `input_guardrail` $\to$ `[if safe: retrieve -> answer -> output_guardrail -> END] | [if blocked: guardrail_block -> END]`
   - Multi-lingual support: English, Nepali, and Romanized Nepali.

---

## 📡 API Endpoints Reference

| Method | Endpoint | Auth Required | Description |
|---|---|---|---|
| `GET` | `/` | None | Operational status and API metadata. |
| `GET` | `/health` | None | Health probe (API, Redis status, Qdrant readiness, Guardrails & Security status). |
| `POST` | `/chat` | Optional (`CHAT_API_KEY`) | Synchronous question answering; returns complete answer JSON with citations. |
| `POST` | `/chat/stream` | Optional (`CHAT_API_KEY`) | **Real-Time Token Streaming (SSE)**: Streams tokens live via Server-Sent Events. |
| `GET` | `/chat/stream` | Optional (`CHAT_API_KEY`) | **Browser EventSource Streaming**: Stream live tokens via browser URL or JS `EventSource`. |
| `POST` | `/rag/upload` | 🔒 **Admin API Key** | **Upload File for RAG**: Index `.pdf`, `.docx`, `.csv`, `.json`, `.txt`, `.md`. |
| `PUT` | `/rag/documents/{filename}` | 🔒 **Admin API Key** | **Update RAG File**: Replace an existing document and re-index its vectors. |
| `DELETE` | `/rag/documents/{filename}` | 🔒 **Admin API Key** | **Delete RAG File**: Remove document from storage and delete its vector points from Qdrant. |
| `GET` | `/rag/documents` | Rate-Limited | **List RAG Documents**: View all indexed files, sizes, and timestamps. |
| `POST` | `/rag/upload-batch` | 🔒 **Admin API Key** | **Batch Upload**: Upload multiple files simultaneously. |
| `POST` | `/rag/test-retrieval` | Rate-Limited | **Test RAG Search**: Search indexed vectors directly without calling the LLM. |
| `POST` | `/admin/ingest/url` | 🔒 **Admin API Key** | **Scrape URL (SSRF Protected)**: Crawl live webpage and index into Qdrant. |
| `GET` | `/chat/{session_id}/history` | Optional (`CHAT_API_KEY`) | Retrieve complete dialogue history stored in Redis. |
| `DELETE` | `/chat/{session_id}` | Optional (`CHAT_API_KEY`) | Clear conversation history from Redis. |
| `GET` | `/docs` | None | Interactive Swagger UI documentation with **Authorize 🔒** dialog. |

---

## 🗑️ How to Delete and Update RAG Files

### 1. Delete a Document
When a corporate document is obsolete (e.g. `old_pricing.pdf`):

**Via PowerShell / cURL:**
```powershell
curl -X DELETE "http://127.0.0.1:8000/rag/documents/old_pricing.pdf" `
     -H "X-API-Key: synk-admin-secret-2026"
```

**Response:**
```json
{
  "status": "success",
  "filename": "old_pricing.pdf",
  "file_deleted": true,
  "vectors_deleted": true,
  "message": "Document 'old_pricing.pdf' and its vector embeddings were successfully deleted from RAG knowledge base."
}
```

---

### 2. Update / Replace an Existing Document
When a document has new terms, pricing, or policies:

**Via PowerShell / cURL:**
```powershell
curl -X PUT "http://127.0.0.1:8000/rag/documents/company_policy.pdf" `
     -H "X-API-Key: synk-admin-secret-2026" `
     -F "file=@documents/company_policy_v2.pdf"
```

**Response:**
```json
{
  "status": "success",
  "filename": "company_policy.pdf",
  "file_type": ".pdf",
  "file_size_kb": 145.2,
  "old_vectors_removed": true,
  "new_chunks_indexed": 18,
  "preview_chunks": [
    "Source: company_policy.pdf (Page 1) Updated 2026 Terms and Service Level Agreements..."
  ],
  "ready_for_chat": true,
  "message": "Document 'company_policy.pdf' successfully updated. Old vectors removed and 18 new chunks indexed."
}
```

---

## 📁 Modular Project Architecture

All Python logic is organized into clean, single-responsibility folders inside [`app/`](file:///d:/synkcorp/app) so that debugging, testing, and adding features is effortless:

```
synkcorp/
├── app/
│   ├── api/                   # HTTP Route Controllers (FastAPI APIRouter)
│   │   ├── chat.py            # /chat, /chat/stream, history & session endpoints
│   │   ├── rag.py             # /rag/upload, update, delete, list, url scraping
│   │   └── health.py          # / and /health status probes
│   ├── core/                  # System Foundations & Policies
│   │   ├── config.py          # Pydantic Settings & environment variables (.env)
│   │   ├── guardrails.py      # Prompt injection blocker, PII redaction & grounding
│   │   └── security.py        # API key auth, IP rate limiter, SSRF & path traversal guards
│   ├── schemas/               # Request/Response Data Models (Pydantic)
│   │   ├── chat.py            # ChatRequest, ChatResponse, Stream events
│   │   └── rag.py             # Upload, update, delete, search schemas
│   ├── services/              # Business Logic & External Engines
│   │   ├── chatbot.py         # LangGraph customer support state graph & streaming
│   │   ├── rag.py             # Qdrant vector store indexing, search & point deletion
│   │   ├── memory.py          # Redis conversational memory & in-memory fallback
│   │   └── ingest.py          # Multi-format parser (PDF, DOCX, CSV, JSON, Web)
│   └── main.py                # FastAPI app factory, CORS, lifespan hooks
├── documents/                 # Corporate documents repository for RAG
├── qdrant_data/               # Persistent local Qdrant vector store
├── api.py                     # Root compatibility runner (uvicorn api:app --reload)
├── main.py                    # Root entrypoint (python main.py)
├── ingest.py                  # CLI ingest tool (python ingest.py)
├── requirements.txt           # Project dependencies
└── README.md                  # System documentation
```

### 🛠️ Where to Debug & Modify

- **Change Prompts / LLM Logic**: Modify [`app/services/chatbot.py`](file:///d:/synkcorp/app/services/chatbot.py).
- **Configure Vector Search & Qdrant**: Modify [`app/services/rag.py`](file:///d:/synkcorp/app/services/rag.py).
- **Update Safety Filters & PII Rules**: Modify [`app/core/guardrails.py`](file:///d:/synkcorp/app/core/guardrails.py).
- **Manage API Keys, Rate Limits & Security Policies**: Modify [`app/core/security.py`](file:///d:/synkcorp/app/core/security.py).
- **Adjust Redis TTL / Fallback Memory**: Modify [`app/services/memory.py`](file:///d:/synkcorp/app/services/memory.py).
- **Add or Modify API Routes**: Update [`app/api/chat.py`](file:///d:/synkcorp/app/api/chat.py) or [`app/api/rag.py`](file:///d:/synkcorp/app/api/rag.py).
- **Add Request / Response Fields**: Update [`app/schemas/chat.py`](file:///d:/synkcorp/app/schemas/chat.py) or [`app/schemas/rag.py`](file:///d:/synkcorp/app/schemas/rag.py).
- **Add Environment Variables**: Update [`app/core/config.py`](file:///d:/synkcorp/app/core/config.py) and `.env`.

---

### 3. Manage via Swagger UI
Open **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**:
- Navigate to **`RAG Document Processing`**:
  - Use **`PUT /rag/documents/{filename}`** to replace an existing file.
  - Use **`DELETE /rag/documents/{filename}`** to delete a file and its vectors.
  - Use **`GET /rag/documents`** to see your active document repository.
