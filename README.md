# ZChat: grounded PDF and JSON question answering

ZChat answers a JSON list of questions from an uploaded PDF or JSON source. It provides a one-request FastAPI endpoint for the take-home challenge, a session API for multiple sources, and a Streamlit UI. Answers use retrieved source text and citations. The configured answer model defaults to gpt-4o-mini.

## High-level design

~~~mermaid
flowchart LR
  UI[Streamlit UI or API client] --> API[FastAPI]
  API -->|upload| RES[Validate and reserve by SHA-256]
  RES --> PARSE[PDF: Docling or native text<br/>JSON: recursive parser]
  PARSE --> TREE[Document tree and bounded search chunks]
  TREE --> SQL[(SQLite: status and original source)]
  TREE --> EMB[FastEmbed: dense and BM25 vectors]
  EMB --> Q[(Qdrant: vector search)]
  API -->|questions| SEARCH[Batch encode, then search each question concurrently]
  Q --> SEARCH
  SEARCH --> RANK[Fuse results and rerank]
  RANK --> EVIDENCE[Load READY source text from SQLite<br/>expand within token budget]
  SQL --> EVIDENCE
  EVIDENCE --> GATE[Answerability check]
  GATE --> LLM[gpt-4o-mini draft and evidence review]
  LLM --> OUT[Validate citations and return ordered JSON]
~~~

**Upload path.** FastAPI validates each file and reserves a document record by hash, so duplicates in the same chat do not need reprocessing. PDF pages are parsed with Docling for layout, tables, and OCR; qualifying large PDFs with usable native text take a faster text path. JSON sources are parsed recursively. The app keeps the original document structure, creates search chunks of at most 512 dense-model tokens, saves original source nodes in SQLite, and indexes batched dense and BM25 vectors in Qdrant. A document becomes READY only after indexing succeeds.

**Answer path.** Question embeddings are batched. Independent questions then run concurrently through chat-scoped dense and BM25 search, result fusion, reranking, and a bounded evidence window. Weak evidence is rejected before calling OpenAI. For supported candidates, gpt-4o-mini drafts an answer and a second model call checks claims in both the answer and comments against source text. The app validates citation IDs before returning results in question order. An unsupported answer stays explicitly not found.

**Async behavior.** The UI sends uploads in the background. After the bytes transfer, the API returns HTTP 202 with a document ID; the UI polls PENDING, PARSING, INDEXING, then READY or FAILED. The user can keep using the UI while parsing runs. This does not make OCR faster. Up to two background ingestions, four retrieval pipelines, and eight LLM calls are admitted per API process. The one-request challenge endpoint is synchronous because it must return completed answers.

## Why this pipeline

LangGraph is useful for stateful agents with branching, loops, and tools. This service has a fixed upload-index-answer path, so an agent graph would add orchestration without simplifying the current behavior. LangChain or LlamaIndex could provide loaders and retrievers, but we need explicit control over file limits, document status, page/JSON provenance, chat isolation, bounded chunks, concurrency, and failure recovery. The small components in app/ own those boundaries directly. The tradeoff is that we maintain more integration code ourselves; a framework could become valuable if the product adds many source connectors or agent workflows.

The retrieval design also goes beyond a basic dense-vector search: dense embeddings catch paraphrases, BM25 catches exact names and codes, and a local cross-encoder reranks the shortlist. Original source text is reloaded from SQLite rather than trusted from vector payloads. This is a design choice for traceability and grounding, not a claim that it outperforms every standard retrieval framework. The included evaluation fixture is too small to establish that.

## Run with Docker Compose

1. Install Docker Desktop or Docker Engine with Compose.
2. From the repository root, copy the example environment file and put your OpenAI key in the ignored .env file. Do not commit it. The challenge PDF's key is not stored in this repository.

   ~~~sh
   cp .env.example .env
   # Edit .env and set OPENAI_API_KEY=...
   ~~~

3. Build and start the three containers:

   ~~~sh
   docker compose up -d --build
   docker compose ps
   ~~~

4. Wait until api, ui, and qdrant show healthy. Check API dependencies:

   ~~~sh
   curl -fsS http://localhost:8000/ready
   ~~~

5. Open the UI at http://localhost:8501. Interactive API docs are at http://localhost:8000/docs. To stop the app without deleting its volumes, run docker compose down.

The first start downloads local embedding, reranking, and Docling models, so readiness may take several minutes. The API can start and index documents without an OpenAI key; supported questions then report LLM_NOT_CONFIGURED. SQLite metadata, Qdrant vectors, and model caches persist in separate Compose volumes. Ports bind to localhost; this setup has no public authentication.

## Use the UI

1. Open http://localhost:8501. A new empty chat is created automatically.
2. In **Documents**, choose evaluation/policy.pdf and evaluation/source.json, then click **Upload & index**. These are small demonstration sources, not the assessment's sample PDF.
3. Watch each document under **Document status**. The UI refreshes while processing; wait for both to show READY. If a document shows FAILED, open its status for the error code and check API logs.
4. In **Questionnaire**, upload evaluation/questionnaire.json and click **Generate batch answers**.
5. Review the rows and click **Download answers JSON**. Expect AWS for cloud, 4 hours for RTO, $7,500 for monthly support, and Data-Not-Found for the CEO. The rows keep input order and contain id, question, answer, comments, and confidence.
6. In **Chat**, ask an individual question such as “What is the RTO?” and inspect the page citation. Selecting a citation points the document preview to its source.
7. Click **New chat** to start with no sources. Previous chats remain isolated in the backend.

The document upload accepts PDF or JSON **sources**. The questionnaire upload accepts a separate JSON **questions file**. A READY source is required before answering. Missing question IDs are generated; provided IDs are preserved.

## Watch live logs

Run these from the repository root in another terminal while using the UI:

~~~sh
docker compose logs -f --tail=100 api
# If the UI cannot connect or render, inspect its container:
docker compose logs -f --tail=100 ui
# If the API is not ready, inspect Qdrant:
docker compose logs -f --tail=100 qdrant
~~~

Application events in the API log are JSON. During an upload, look for document_ingested with parser_route (docling, native_text, or json), page_count, retrieval_nodes, parse_ms, and ingestion_ms. A failed upload logs document_ingestion_failed with a sanitized error_code. Each answered question logs question_answered with retrieval_ms, rerank_ms, generation_ms, total_ms, supported, and error_code. Logs omit source text, prompts, questions, and API keys. The /ready endpoint checks backend dependencies and reports whether an answer provider is configured; it does not call OpenAI.

## API example

The challenge-compatible endpoint accepts a document and a JSON questions file in one multipart request:

~~~sh
curl -sS -X POST 'http://localhost:8000/api/v1/qa?output=questionnaire' \
  -F 'document=@evaluation/policy.pdf' \
  -F 'questions=@evaluation/questionnaire.json' \
  -o answers.json
~~~

The source document can be PDF or JSON. Questions can be a JSON array of strings, an array of records with question and optional id, or either array inside a questions object. The questionnaire output is an ordered array containing exactly id, question, answer, comments, and confidence. Confidence (high, medium, low) describes evidence strength, not a calibrated probability. Unsupported answers use Data-Not-Found and low confidence; provider or processing failures use Error and a sanitized code rather than masquerading as missing facts. Existing input answer, comments, and confidence fields are discarded, never used as evidence.

Omit the output query parameter for the detailed API response with citations, support status, errors, and stage timings. The session endpoints at /api/v1/chats let clients upload more sources and ask more questions without reuploading previous documents; explore them at /docs. The default document upload is synchronous. Add ?background=true for HTTP 202 and poll GET /api/v1/chats/{chat_id}/documents until READY or FAILED.

## Tests and limits

For local tests, install Python 3.11-3.13 and uv (https://docs.astral.sh/uv/):

~~~sh
uv sync --frozen --group ui
uv run pytest -q
uv run ruff check app tests scripts streamlit_app.py
~~~

The standard suite covers core logic and mocked FastAPI endpoints, including PDF/JSON inputs, validation, deduplication, citations, chat isolation, batch order/concurrency, and recovery. For a real Docling/FastEmbed/Qdrant integration test with mocked OpenAI HTTP:

~~~sh
docker compose up -d qdrant
RUN_LIVE_TESTS=1 uv run pytest -q tests/test_live.py
~~~

Run uv run python scripts/evaluate.py for retrieval metrics on a 17-question synthetic fixture without paid model calls. Add --generate to evaluate generated answers and citations using the configured API key. This small fixture is a smoke test, not a representative accuracy benchmark.

Defaults include 50 MB per file, 1,500 PDF pages, 50 questions per batch, and a 3,600-second parse timeout; see .env.example and app/config.py. These are admission limits, not throughput guarantees. OCR and complex layouts can be slow: a local 21-page mixed text-and-images guide took 54 seconds to parse warm, about 50 seconds of which was OCR. Background jobs live in one API process; a restart marks unfinished work FAILED for retry. Multi-instance coordination, durable workers, authentication, and document expiration are outside this local submission. Confidence is not calibrated, and diagrams without readable text may remain unanswerable.
