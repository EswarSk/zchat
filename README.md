# ZChat: document question answering

ZChat answers a JSON list of questions about a PDF or JSON document through a FastAPI endpoint. It also offers a session API for accumulating documents and a small Streamlit UI. Answers use retrieved source text and citations; unsupported questions return Data-Not-Found in questionnaire output. The default answer model is gpt-4o-mini.

## Run

Requirements: Docker with Compose and an OpenAI API key for generated answers. The API can start and index documents without a key, but answering then reports LLM_NOT_CONFIGURED. The first start downloads local embedding, reranking, and Docling models and can take several minutes.

~~~sh
cp .env.example .env
# Add OPENAI_API_KEY to .env. Never commit this file.
docker compose up --build
~~~

- UI: http://localhost:8501
- Interactive API docs: http://localhost:8000/docs
- API dependency check: http://localhost:8000/ready

Compose starts one FastAPI process, Qdrant, and Streamlit. SQLite metadata, Qdrant vectors, and model caches persist in separate Docker volumes. Ports bind to localhost. This is a local submission, not an authenticated public deployment.

## Try the challenge flow

The single-request endpoint accepts a source document (PDF or JSON) and a JSON questions file. The repository includes small fixtures: evaluation/policy.pdf, evaluation/source.json, and evaluation/questionnaire.json.

~~~sh
curl -sS -X POST 'http://localhost:8000/api/v1/qa?output=questionnaire' \
  -F 'document=@evaluation/policy.pdf' \
  -F 'questions=@evaluation/questionnaire.json' \
  -o answers.json
~~~

answers.json is an array in input order. Each row has exactly id, question, answer, comments, and confidence. Supplied IDs are retained; missing IDs are generated. Comments explain the answer and name its cited source. Confidence is a qualitative evidence label (high, medium, or low), not a probability. An unsupported answer is Data-Not-Found with low confidence; a provider or processing error is Error with a sanitized code in comments.

Omit the output query parameter for the detailed response: chat ID, each question's answer or nullable missing answer, citations, support status, per-question error, and stage timings. Questions can be an array of strings, an array of records with question and optional id, or either array inside a questions object. Existing input answer, comments, and confidence fields are ignored and never used as evidence.

In the UI, upload the PDF or JSON source, wait for READY, then upload the questions JSON under **Questionnaire** and download the answer JSON. The included fixture should support AWS and a four-hour RTO, while a CEO question should return Data-Not-Found. Individual questions and page citations are also available in the chat pane.

## Background upload and session API

A chat can accumulate PDF and JSON sources. The UI uses background uploads and polls status every three seconds.

~~~sh
curl -sS -X POST http://localhost:8000/api/v1/chats
# Set CHAT_ID to the returned chat_id.
curl -sS -X POST "http://localhost:8000/api/v1/chats/$CHAT_ID/documents?background=true" \
  -F 'documents=@evaluation/policy.pdf'
curl -sS "http://localhost:8000/api/v1/chats/$CHAT_ID/documents"
curl -sS -X POST "http://localhost:8000/api/v1/chats/$CHAT_ID/questions?output=questionnaire" \
  -F 'questions=@evaluation/questionnaire.json'
~~~

The background upload returns HTTP 202 and a document ID after transfer. Poll until READY or FAILED; failures expose a safe error code. The default document upload and /api/v1/qa endpoints are synchronous because they return completed work. A repeated file in the same chat is deduplicated by SHA-256; a failed upload can be retried. New chats have isolated sources.

## Design

~~~text
PDF/JSON upload -> validate and reserve in SQLite -> parse -> document tree
    -> bounded retrieval chunks -> batched dense + BM25 vectors -> Qdrant

Question list -> batched question vectors -> concurrent per-question search
    -> dense/sparse fusion -> local reranker -> bounded source context
    -> answerability check -> gpt-4o-mini draft -> evidence review
    -> citation validation -> ordered JSON results
~~~

- **Parsing and chunking:** JSON is parsed recursively. Docling extracts PDF layout, tables, text, and OCR where needed. PDFs of at least 200 pages use a faster native-text path only if at least 95% of pages have sufficient native text and no short image pages. Original structural nodes are retained. Search chunks are capped at 512 dense-model tokens; tables split by row groups with repeated headers. Citations retain PDF page numbers or JSON key paths.
- **Retrieval and grounding:** FastEmbed runs BAAI/bge-small-en-v1.5 dense embeddings, Qdrant/bm25 sparse keyword vectors, and a local cross-encoder reranker. Qdrant searches only the chat's READY documents; source text is reloaded from SQLite rather than trusted from vector payloads. A weak-evidence gate avoids unnecessary OpenAI calls. The model receives bounded raw evidence, and a second evidence check reviews claims in both answer and comments. Citation IDs are validated before return.
- **Concurrency and failures:** Batch questions run concurrently, with shared limits of four retrieval pipelines and eight LLM calls. Embeddings are batched; local model inference is serialized to avoid CPU oversubscription. At most two background ingestions run at once; excess work receives HTTP 429 INGESTION_BUSY. PDF conversion runs in a child process with a timeout. Interrupted or failed ingestion is marked FAILED, and only READY documents can be searched. Upload and question limits, explicit error codes, and structured JSON logs cover the principal failure paths.

The API does not use LangChain or LlamaIndex. Its parser, indexer, retriever, and generator are separate Python components providing the equivalent document-QA pipeline with explicit, testable boundaries.

## Verify

For local tests, install Python 3.11-3.13 and uv (https://docs.astral.sh/uv/):

~~~sh
uv sync --frozen --group ui
uv run pytest -q
uv run ruff check app tests scripts streamlit_app.py
~~~

The standard suite covers core logic and FastAPI endpoints with controlled parsing, local-model, and LLM doubles. It exercises PDF/JSON handling, validation, deduplication, citations, session isolation, batch order and concurrency, and recovery. An opt-in integration test runs real Docling, FastEmbed, and Qdrant while mocking OpenAI HTTP:

~~~sh
docker compose up -d qdrant
RUN_LIVE_TESTS=1 uv run pytest -q tests/test_live.py
~~~

Run uv run python scripts/evaluate.py for free retrieval metrics on a 17-question synthetic fixture. Add --generate to evaluate generated answers and citations using the configured API key. This fixture is a smoke test, not a representative benchmark or production accuracy claim.

## Boundaries

- Defaults: 50 MB per file, 1,500 PDF pages, 50 questions per batch, 2,000 characters per question, and a 3,600-second PDF parse timeout. These are admission and safety limits, not throughput guarantees. See .env.example and app/config.py.
- OCR and complex layouts can be slow. In one local run, a 21-page mixed text-and-images guide parsed in 54 seconds warm; about 50 seconds was OCR. Background upload keeps the UI responsive but does not shorten OCR.
- Background jobs live in one API process. A restart marks unfinished documents failed for retry. Durable workers, multi-instance coordination, authentication, and document expiration are outside this local submission.
- Confidence is not calibrated. The reranker cutoff and retrieval results were checked on a small synthetic fixture; broad PDF and answer-faithfulness evaluation remains necessary before production use. Diagrams without readable text may not be answerable.

The challenge PDF's API key is not used or stored in this repository. Supply your own key through the ignored .env file.
