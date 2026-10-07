"""A session-scoped reviewer UI; all ingestion and QA stay in the API."""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import httpx
import streamlit as st

API_URL = os.environ.get("ZCHAT_API_URL", "http://localhost:8000").rstrip("/")


class APIError(Exception):
    pass


def api(method, path, **kwargs):
    try:
        response = httpx.request(
            method,
            f"{API_URL}{path}",
            timeout=kwargs.pop("timeout", httpx.Timeout(900, connect=5)),
            **kwargs,
        )
    except httpx.TimeoutException as exc:
        raise APIError(
            "The API timed out. An upload may still be processing; refresh document status."
        ) from exc
    except httpx.RequestError as exc:
        raise APIError("Cannot reach the API. Check that the backend is running.") from exc
    try:
        payload = response.json()
    except ValueError as exc:
        raise APIError(
            f"The API returned an invalid response (HTTP {response.status_code})."
        ) from exc
    if not response.is_success:
        error = payload.get("error", {})
        raise APIError(
            f"{error.get('code', response.status_code)}: {error.get('message', 'Request failed')}"
        )
    return payload


def new_chat():
    # Create first: a failed reset must preserve the current session and its previews.
    chat_id = api("POST", "/api/v1/chats")["chat_id"]
    st.session_state.update(
        chat_id=chat_id, messages=[], sources={}, upload_notices=[], batch_result=None
    )
    st.session_state.pop("preview_id", None)
    st.session_state.pop("citation_location", None)
    st.session_state.pop("upload_job", None)


@st.cache_resource
def upload_executor():
    return ThreadPoolExecutor(max_workers=2)


def upload_documents(chat_id, uploads):
    results = []
    for filename, data in uploads:
        kind = "application/pdf" if filename.lower().endswith(".pdf") else "application/json"
        try:
            item = api(
                "POST",
                f"/api/v1/chats/{chat_id}/documents",
                files={"documents": (filename, data, kind)},
                timeout=httpx.Timeout(7200, connect=5),
            )["documents"][0]
            results.append((filename, data, item, None))
        except APIError as exc:
            results.append((filename, data, None, str(exc)))
    return results


def select_citation(document_id, label):
    st.session_state.preview_id = document_id
    st.session_state.citation_location = label


@st.fragment(run_every="3s")
def watch_documents(chat_id, documents):
    """Refresh the whole UI only when processing state changes."""
    st.info("Documents are processing. Status updates automatically; answers unlock when READY.")
    job = st.session_state.get("upload_job")
    if job is not None and job.done():
        st.rerun()
    try:
        current = api("GET", f"/api/v1/chats/{chat_id}/documents")["documents"]
    except APIError as exc:
        st.warning(str(exc))
        return
    if current != documents:
        st.rerun()


def render_answer(result, index):
    if result.get("error"):
        error = result["error"]
        st.error(f"{error['code']}: {error['message']}")
    elif result.get("supported"):
        st.markdown(result["answer"])
    else:
        st.info("No supporting evidence was found in this chat's documents.")
    for citation_index, citation in enumerate(result.get("citations", [])):
        page = citation["page_start"]
        end = citation["page_end"]
        location = f"page {page}" if page == end else f"pages {page}–{end}"
        if page is None:
            location = " > ".join(citation["heading_path"]) or "JSON source"
        label = f"{citation['filename']} · {location}"
        st.button(
            label,
            key=f"citation_{index}_{citation_index}",
            width="stretch",
            on_click=select_citation,
            args=(citation["document_id"], label),
        )
    if result.get("timing"):
        with st.expander("Response details"):
            st.json(result)


def main():
    st.set_page_config(page_title="ZChat · Document QA", page_icon="📄", layout="wide")
    title, reset = st.columns([5, 1], vertical_alignment="center")
    with title:
        st.title("ZChat")
        st.caption("Upload sources on the left. Ask grounded questions on the right.")
    try:
        with reset:
            if st.button("New chat", width="stretch"):
                new_chat()
                st.rerun()
        readiness = api("GET", "/ready")
        if "chat_id" not in st.session_state:
            new_chat()
        chat_id = st.session_state.chat_id
        job = st.session_state.get("upload_job")
        if job is not None and job.done():
            st.session_state.pop("upload_job")
            for filename, data, item, error in job.result():
                if error:
                    st.session_state.upload_notices.append(f"{filename}: {error}")
                else:
                    st.session_state.sources[item["document_id"]] = data
                    st.session_state.preview_id = item["document_id"]
                    suffix = " (already uploaded)" if item["duplicate"] else ""
                    st.session_state.upload_notices.append(
                        f"{item['filename']}: upload received{suffix}"
                    )
        documents = api("GET", f"/api/v1/chats/{chat_id}/documents")["documents"]
    except APIError as exc:
        st.error(str(exc))
        if st.button("Retry connection"):
            st.rerun()
        st.stop()

    st.caption(f"Session: {chat_id}")
    if not readiness["answer_provider_configured"]:
        st.warning(
            "Answer generation needs OPENAI_API_KEY in the backend .env. "
            "Uploads, previews, and unsupported-question checks are available."
        )
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Documents")
        with st.form(f"upload_{chat_id}", clear_on_submit=True):
            uploads = st.file_uploader(
                "PDF or JSON sources",
                type=["pdf", "json"],
                accept_multiple_files=True,
                max_upload_size=50,
                key=f"files_{chat_id}",
            )
            submitted = st.form_submit_button(
                "Upload & index", width="stretch", disabled="upload_job" in st.session_state
            )
        if submitted:
            st.session_state.upload_notices = []
            if not uploads:
                st.warning("Choose at least one document first.")
            if uploads:
                st.session_state.upload_job = upload_executor().submit(
                    upload_documents, chat_id, [(u.name, u.getvalue()) for u in uploads]
                )
                st.rerun()
        for notice in st.session_state.upload_notices:
            st.caption(notice)
        if st.button("Refresh document status"):
            st.rerun()
        if "upload_job" in st.session_state or any(
            d["status"] in {"PENDING", "PARSING", "INDEXING"} for d in documents
        ):
            watch_documents(chat_id, documents)
        if not documents:
            st.info("Add a PDF or JSON document to begin. New chats start empty.")
        else:
            by_id = {document["id"]: document for document in documents}
            if st.session_state.get("preview_id") not in by_id:
                st.session_state.preview_id = documents[0]["id"]
            selected = st.selectbox(
                "Preview document",
                list(by_id),
                format_func=lambda doc_id: (
                    f"{by_id[doc_id]['filename']} · {by_id[doc_id]['status']}"
                ),
                key="preview_id",
                on_change=lambda: st.session_state.pop("citation_location", None),
            )
            with st.expander(f"Document status · {len(documents)} in this chat"):
                for doc in documents:
                    st.text(f"{doc['filename']}: {doc['status']}")
                    if doc.get("error_message"):
                        st.error(doc["error_message"])
            if st.session_state.get("citation_location"):
                st.caption(f"Selected citation: {st.session_state.citation_location}")
            data = st.session_state.sources.get(selected)
            if data is None:
                st.info("Upload this source again to preview it; the API will deduplicate it.")
            elif by_id[selected]["content_type"] == "application/pdf":
                st.pdf(data, height=620, key=f"pdf_{selected}", alt=by_id[selected]["filename"])
            else:
                st.code(data.decode("utf-8-sig", errors="replace"), language="json", height=620)

    with right:
        st.subheader("Questionnaire")
        st.caption("Upload questions as JSON. Download answers in the sample's five-column format.")
        st.caption("Use question strings or records with question and an optional id.")
        if documents and not any(d["status"] == "READY" for d in documents):
            if any(d["status"] == "FAILED" for d in documents):
                st.error("Source processing failed. Check Document status for the error.")
            else:
                st.info("Waiting for source documents to finish parsing and indexing.")
        with st.form(f"questionnaire_{chat_id}"):
            questions_file = st.file_uploader(
                "Questions JSON", type=["json"], max_upload_size=1, key=f"questions_{chat_id}"
            )
            generate = st.form_submit_button(
                "Generate batch answers",
                disabled=not any(d["status"] == "READY" for d in documents),
                width="stretch",
            )
        if generate:
            if questions_file is None:
                st.warning("Choose a questions JSON file first.")
            else:
                try:
                    with st.spinner("Answering your questions from the uploaded sources…"):
                        records = api(
                            "POST",
                            f"/api/v1/chats/{chat_id}/questions?output=questionnaire",
                            files={
                                "questions": (
                                    questions_file.name,
                                    questions_file.getvalue(),
                                    "application/json",
                                )
                            },
                        )
                    st.session_state.batch_result = {
                        "filename": questions_file.name,
                        "records": records,
                    }
                except APIError as exc:
                    st.error(str(exc))
        batch = st.session_state.get("batch_result")
        if batch:
            st.caption(
                f"Last completed batch: {batch['filename']} · {len(batch['records'])} questions"
            )
            if any(
                row["answer"] == "Error" and row["confidence"] == "low" for row in batch["records"]
            ):
                st.warning("Some questions failed. Error rows include the reason; retry the batch.")
            st.dataframe(batch["records"], hide_index=True, width="stretch")
            st.download_button(
                "Download answers JSON",
                data=json.dumps(batch["records"], ensure_ascii=False, indent=2),
                file_name="answers.json",
                mime="application/json",
                on_click="ignore",
            )
            with st.expander("Answers JSON"):
                st.json(batch["records"])
        st.subheader("Chat")
        st.caption(
            "Follow-ups use your recent questions for context. Answers come from your documents."
        )
        with st.container(height=620, border=True):
            if not st.session_state.messages:
                st.info("Ask a question about your sources. Answers include source citations.")
            for index, message in enumerate(st.session_state.messages):
                with st.chat_message("user"):
                    st.markdown(message["question"])
                with st.chat_message("assistant"):
                    render_answer(message["result"], index)
        question = st.chat_input("Ask about your documents…", max_chars=2000)
        if question and question.strip():
            try:
                with st.spinner("Finding evidence and preparing an answer…"):
                    result = api(
                        "POST",
                        f"/api/v1/chats/{chat_id}/questions",
                        json={
                            "questions": [question],
                            "previous_questions": [
                                m["result"].get("resolved_question") or m["question"]
                                for m in st.session_state.messages[-5:]
                            ],
                        },
                    )["results"][0]
            except APIError as exc:
                result = {"error": {"code": "REQUEST_FAILED", "message": str(exc)}}
            st.session_state.messages.append({"question": question, "result": result})
            st.rerun()


if __name__ == "__main__":
    main()
