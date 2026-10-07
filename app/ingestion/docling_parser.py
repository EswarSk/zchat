import asyncio
import io
import logging
import multiprocessing
import os
import threading
from pathlib import Path
from time import monotonic

from pypdf import PdfReader
from pypdfium2 import PdfDocument, PdfImage

from app.api.errors import AppError
from app.domain.models import DocumentNode
from app.ingestion.document_tree import ParsedBlock, build_tree


def validate_pdf(source: bytes | Path, max_pages: int) -> int:
    try:
        stream = source.open("rb") if isinstance(source, Path) else io.BytesIO(source)
        with stream:
            if stream.read(5) != b"%PDF-":
                raise AppError(422, "DOCUMENT_PARSE_FAILED", "Invalid PDF signature")
            stream.seek(0)
            reader = PdfReader(stream)
            if reader.is_encrypted:
                raise AppError(422, "DOCUMENT_PARSE_FAILED", "Encrypted PDFs are not supported")
            count = len(reader.pages)
            if count == 0:
                raise AppError(422, "DOCUMENT_PARSE_FAILED", "PDF has no pages")
            if count > max_pages:
                raise AppError(413, "PDF_TOO_MANY_PAGES", "PDF exceeds configured page limit")
            return count
    except AppError:
        raise
    except Exception as exc:
        raise AppError(422, "DOCUMENT_PARSE_FAILED", "PDF could not be read") from exc


def native_text_document(path: Path, chat_id: str, document_id: str, count: int):
    """Use page-scoped native text when full layout inference is impractical."""
    pdf = PdfDocument(path)
    try:
        if len(pdf) != count:
            raise AppError(422, "DOCUMENT_PARSE_FAILED", "PDF page counts disagree")
        pages = []
        rich_pages = 0
        for index in range(count):
            page = pdf[index]
            try:
                text_page = page.get_textpage()
                try:
                    text = text_page.get_text_range().replace("\r\n", "\n")
                finally:
                    text_page.close()
                if len(text.strip()) >= 100:
                    rich_pages += 1
                elif any(isinstance(obj, PdfImage) for obj in page.get_objects()):
                    # A short page with a raster image may contain facts requiring OCR.
                    return None
                pages.append(text)
            finally:
                page.close()
            # Stop scanning as soon as the native-text route cannot qualify.
            if rich_pages + count - index - 1 < 0.95 * count:
                return None
        # A few title or divider pages can be short; mostly image pages still need OCR.
        if rich_pages < 0.95 * count:
            return None
        return build_tree(
            chat_id,
            document_id,
            [
                ParsedBlock("passage", text, index, index, metadata={"parser": "pypdfium2"})
                for index, text in enumerate(pages, start=1)
            ],
        )
    finally:
        pdf.close()


def convert_document(document, chat_id, document_id, pdf_path: Path | None = None):
    """The only adapter that knows Docling's item types."""
    from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownParams
    from docling_core.types.doc import (
        DocItemLabel,
        ListItem,
        PictureItem,
        SectionHeaderItem,
        TableItem,
        TextItem,
    )

    # Column-width padding is presentation, not evidence, and can exhaust model tokens.
    serializer = MarkdownDocSerializer(doc=document, params=MarkdownParams(compact_tables=True))
    blocks = []
    picture_pages = set()
    for item, _depth in document.iterate_items():
        pages = [p.page_no for p in getattr(item, "prov", [])]
        page_start, page_end = (min(pages), max(pages)) if pages else (None, None)
        metadata = {"source_ref": item.self_ref, "label": str(getattr(item, "label", ""))}
        if isinstance(item, PictureItem):
            picture_pages.update(pages)
        if isinstance(item, SectionHeaderItem):
            kind, text, level = "section", item.text, item.level
        elif isinstance(item, TableItem):
            kind, text, level = "table", serializer.serialize(item=item).text, 0
        elif isinstance(item, ListItem):
            kind, text, level = "list", item.text, 0
            metadata["marker"] = item.marker
        elif isinstance(item, TextItem):
            if item.label in (DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER):
                continue
            kind = "section" if item.label == DocItemLabel.TITLE else "passage"
            text, level = item.text, 1
        else:
            continue
        blocks.append(ParsedBlock(kind, text, page_start, page_end, level, metadata))
    nodes = build_tree(chat_id, document_id, blocks)
    if pdf_path is not None and picture_pages:
        # Layout detection can misclassify native text as a picture (including headings).
        # Keep a page-scoped source fallback; normal segmentation still bounds model inputs.
        reader = PdfReader(pdf_path)
        for page in sorted(picture_pages):
            text = reader.pages[page - 1].extract_text() or ""
            if text.strip():
                nodes.append(
                    DocumentNode(
                        chat_id=chat_id,
                        document_id=document_id,
                        parent_id=nodes[0].id,
                        node_type="passage",
                        level=1,
                        text=text,
                        page_start=page,
                        page_end=page,
                        ordinal=len(nodes),
                        metadata={"parser": "pypdf", "reason": "native_text_on_picture_page"},
                    )
                )
                nodes[0].page_start = min(nodes[0].page_start or page, page)
                nodes[0].page_end = max(nodes[0].page_end or page, page)
    return nodes


def _parse_worker(pipe, path: str, chat_id: str, document_id: str, settings_dict: dict):
    # PDF validation also runs here: a malicious PDF must not block the API process.
    try:
        logging.getLogger().setLevel(logging.ERROR)
        # Docling runs inference on worker threads. Bound their native thread pools too.
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[variable] = str(settings_dict["model_threads"])
        count = validate_pdf(Path(path), settings_dict["max_pdf_pages"])
        # Full page layout and OCR can take tens of minutes on book-length PDFs.
        # Native text keeps every page searchable and preserves citation pages.
        nodes = (
            native_text_document(Path(path), chat_id, document_id, count) if count >= 200 else None
        )
        if nodes is not None:
            if len(nodes) > settings_dict["max_document_nodes"]:
                raise AppError(
                    422, "DOCUMENT_PARSE_FAILED", "Document exceeds configured node limit"
                )
            pipe.send((True, nodes, count))
            return
        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            HeadingHierarchyOptions,
            PdfPipelineOptions,
            RapidOcrOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption

        options = PdfPipelineOptions(
            heading_hierarchy_options=HeadingHierarchyOptions(enabled=True),
            generate_parsed_pages=True,
            document_timeout=settings_dict["pdf_parse_timeout_seconds"],
            ocr_options=RapidOcrOptions(lang=["en"], backend="onnxruntime"),
            accelerator_options=AcceleratorOptions(num_threads=settings_dict["model_threads"]),
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=options),
            }
        )
        result = converter.convert(path, raises_on_error=True)
        # Do not silently accept partially parsed documents.
        if result.status.value != "success":
            raise AppError(422, "DOCUMENT_PARSE_FAILED", "PDF conversion was incomplete")
        nodes = convert_document(result.document, chat_id, document_id, Path(path))
        if len(nodes) > settings_dict["max_document_nodes"]:
            raise AppError(422, "DOCUMENT_PARSE_FAILED", "Document exceeds configured node limit")
        pipe.send((True, nodes, count))
    except AppError as exc:
        pipe.send((False, exc.status, exc.code, exc.message))
    except Exception:
        pipe.send((False, 422, "DOCUMENT_PARSE_FAILED", "PDF conversion failed"))
    finally:
        pipe.close()


class DoclingParser:
    def __init__(self, settings):
        self.settings = settings

    async def parse_async(self, path: Path, chat_id: str, document_id: str):
        cancelled = threading.Event()
        task = asyncio.create_task(
            asyncio.to_thread(
                self.parse,
                path,
                chat_id,
                document_id,
                cancelled,
            )
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled.set()
            # Keep the upload's semaphore and tempfile until the subprocess is stopped.
            await asyncio.gather(task, return_exceptions=True)
            raise

    def parse(self, path: Path, chat_id: str, document_id: str, cancelled=None):
        context = multiprocessing.get_context("spawn")
        receive, send = context.Pipe(duplex=False)
        process = context.Process(
            target=_parse_worker,
            args=(
                send,
                str(path),
                chat_id,
                document_id,
                self.settings.model_dump(mode="json"),
            ),
        )
        process.start()
        send.close()
        try:
            deadline = monotonic() + self.settings.pdf_parse_timeout_seconds
            while True:
                if cancelled is not None and cancelled.is_set():
                    raise AppError(499, "PDF_PARSE_CANCELLED", "PDF parsing was cancelled")
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise AppError(
                        504, "PDF_PARSE_TIMEOUT", "PDF parsing exceeded configured timeout"
                    )
                if receive.poll(min(remaining, 0.1)):
                    break
            result = receive.recv()
            if not result[0]:
                raise AppError(result[1], result[2], result[3])
            return result[1], result[2]
        except EOFError as exc:
            raise AppError(422, "DOCUMENT_PARSE_FAILED", "PDF parser exited unexpectedly") from exc
        finally:
            receive.close()
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()
