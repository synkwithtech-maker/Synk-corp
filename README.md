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

3. **Defense-in-Depth Guardrails (`guardrails.py`)**:
   - **Prompt Injection & Jailbreak Prevention**: Blocks instruction overrides (`ignore previous instructions`, `DAN`, persona hijacks) before hitting the vector database or LLM.
   - **PII Redaction**: Automatically detects and masks sensitive data (Credit Cards, SSNs, API secrets) to protect customer privacy and prevent memory leaks.
   - **System Prompt Leak Suppression**: Inspects generated output to ensure internal prompt instructions and model delimiters are never regurgitated.
   - **Strict Grounding Enforcement**: Guarantees answers strictly reflect Synk Corp documents, falling back to a safe refusal when information is absent.

4. **Redis Conversation Memory (`memory.py`)**:
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

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Operational status and API metadata. |
| `GET` | `/health` | Health probe (API, Redis status & latency, Qdrant readiness, Guardrails info). |
| `POST` | `/chat` | Synchronous question answering; returns complete answer JSON with citations and latency. |
| `POST` | `/chat/stream` | **Real-Time Token Streaming (SSE)**: Streams tokens live via Server-Sent Events. |
| `GET` | `/chat/stream` | **Browser EventSource Streaming**: Stream live tokens via browser URL or native JS `EventSource`. |
| `POST` | `/rag/upload` | **Upload File for RAG**: Send `.pdf`, `.docx`, `.csv`, `.json`, `.txt`, `.md` to index into Qdrant. |
| `PUT` | `/rag/documents/{filename}` | **Update RAG File**: Replace an existing document and re-index its vectors. |
| `DELETE` | `/rag/documents/{filename}` | **Delete RAG File**: Remove a document from storage and delete its vector points from Qdrant. |
| `GET` | `/rag/documents` | **List RAG Documents**: View all indexed files, sizes, and timestamps. |
| `POST` | `/rag/upload-batch` | **Batch Upload**: Upload multiple files simultaneously. |
| `POST` | `/rag/test-retrieval` | **Test RAG Search**: Search indexed vectors directly without calling the LLM. |
| `POST` | `/admin/ingest/url` | **Scrape URL**: Crawl a live website URL, extract clean text, and index into Qdrant. |
| `GET` | `/chat/{session_id}/history` | Retrieve complete dialogue history stored in Redis for that session. |
| `DELETE` | `/chat/{session_id}` | Clear conversation history from Redis. |
| `GET` | `/docs` | Interactive Swagger UI documentation. |

---

## 🗑️ How to Delete and Update RAG Files

### 1. Delete a Document
When a corporate document is obsolete (e.g. `old_pricing.pdf`):

**Via PowerShell / cURL:**
```powershell
curl -X DELETE "http://127.0.0.1:8000/rag/documents/old_pricing.pdf"
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

### 3. Manage via Swagger UI
Open **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**:
- Navigate to **`RAG Document Processing`**:
  - Use **`PUT /rag/documents/{filename}`** to replace an existing file.
  - Use **`DELETE /rag/documents/{filename}`** to delete a file and its vectors.
  - Use **`GET /rag/documents`** to see your active document repository.
