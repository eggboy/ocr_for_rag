"""GLM-OCR pipeline with optional layout-aware mode.

Uses the `zai-org/GLM-OCR` model — a dedicated 0.9B multimodal OCR model
by Z.ai, served via vLLM (OpenAI-compatible API).

Modes:
    simple (default):
        Sends whole images / PDF pages directly to the GLM-OCR model.

    layout:
        Uses PaddleOCR PP-DocLayoutV3 for document layout detection, crops
        each detected region, sends them to GLM-OCR with region-type-specific
        task prompts in parallel, and reassembles results in reading order.

Both modes use PyMuPDF for high-quality PDF-to-image rendering (no poppler
or pdf2image required).  Layout mode additionally requires ``paddleocr``,
``numpy``, and ``pillow`` (installed via the ``[layout]`` extra).
"""

from __future__ import annotations

import base64
import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pymupdf
from dotenv import load_dotenv

if TYPE_CHECKING:
    from PIL import Image as PILImage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "zai-org/GLM-OCR"
DEFAULT_ENDPOINT = "http://localhost:8000/v1/chat/completions"
REQUEST_TIMEOUT = 300.0
OUTPUT_DIR = Path("./output")
RESPONSE_OUTPUT_PATH = Path("output/glm_response.json")
PDF_RENDER_DPI = 300
MAX_WORKERS = 4

_IMAGE_MIME_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
}

_SUPPORTED_EXTENSIONS = {*_IMAGE_MIME_TYPES.keys(), ".pdf"}

# ---------------------------------------------------------------------------
# GLM-OCR task prompts
# ---------------------------------------------------------------------------
# The model recognises these task-specific prefixes natively.
TASK_TEXT = "Text Recognition:"
TASK_TABLE = "Table Recognition:"
TASK_FORMULA = "Formula Recognition:"

# Layout labels recognised by PP-DocLayoutV3 that contain OCR-worthy content
_OCR_LABELS: set[str] = {
    "text",
    "paragraph_title",
    "doc_title",
    "table",
    "table_title",
    "figure_title",
    "formula",
    "reference",
    "abstract",
    "content",
    "header",
    "footer",
    "footnote",
    "algorithm",
    "seal",
    "chart",
    "sidebar_text",
    "reference_content",
}

# Map layout labels to the best GLM-OCR task prompt
_LABEL_TO_TASK: dict[str, str] = {
    "table": TASK_TABLE,
    "table_title": TASK_TEXT,
    "formula": TASK_FORMULA,
}


def _task_prompt_for_label(label: str) -> str:
    """Return the GLM-OCR task prompt best suited for a layout region label."""
    return _LABEL_TO_TASK.get(label, TASK_TEXT)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _encode_image(file_path: Path) -> tuple[str, str]:
    """Base64-encode an image file and return (base64_data, mime_type)."""
    suffix = file_path.suffix.lower()
    mime = _IMAGE_MIME_TYPES.get(suffix)
    if not mime:
        supported = ", ".join(sorted(_IMAGE_MIME_TYPES.keys()))
        raise ValueError(f"Unsupported image type '{suffix}'. Supported: {supported}")

    b64 = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    return b64, mime


def _render_pdf_page(page: pymupdf.Page, dpi: int = PDF_RENDER_DPI) -> bytes:
    """Render a single PyMuPDF page to PNG bytes."""
    zoom = dpi / 72
    mat = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes(output="png")


def _pdf_pages_to_base64(file_path: Path, dpi: int = PDF_RENDER_DPI) -> list[tuple[str, str]]:
    """Render each page of a PDF to a PNG image and return base64-encoded data."""
    doc = pymupdf.open(file_path)
    results: list[tuple[str, str]] = []
    try:
        for page_num in range(len(doc)):
            png_bytes = _render_pdf_page(doc[page_num], dpi)
            b64 = base64.b64encode(png_bytes).decode("utf-8")
            results.append((b64, "image/png"))
            logger.debug("Rendered PDF page %d/%d", page_num + 1, len(doc))
    finally:
        doc.close()
    return results


def _pdf_pages_to_pil(file_path: Path, dpi: int = PDF_RENDER_DPI) -> list[PILImage.Image]:
    """Render each page of a PDF to a PIL Image via PyMuPDF."""
    from PIL import Image

    doc = pymupdf.open(file_path)
    images: list[PILImage.Image] = []
    try:
        for page_num in range(len(doc)):
            png_bytes = _render_pdf_page(doc[page_num], dpi)
            images.append(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
            logger.debug("Rendered PDF page %d/%d to PIL", page_num + 1, len(doc))
    finally:
        doc.close()
    return images


def _pil_to_base64(img: PILImage.Image, fmt: str = "PNG") -> str:
    """Encode a PIL Image to a base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_payload(
    model: str,
    b64_data: str,
    mime_type: str,
    prompt: str,
) -> dict:
    """Construct the OpenAI-compatible chat completion payload."""
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64_data}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
    }


def _parse_response(result: dict) -> str:
    """Extract the text content from an OpenAI-compatible API response."""
    if choices := result.get("choices"):
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if content:
            return content

    if "content" in result:
        return result["content"]

    logger.warning("Unrecognised response format. Keys: %s", list(result.keys()))
    return ""


def _send_ocr_request(
    endpoint: str,
    headers: dict[str, str],
    model: str,
    b64_data: str,
    mime_type: str,
    prompt: str,
) -> str:
    """Send a single OCR request to the GLM-OCR API and return extracted text."""
    payload = _build_payload(model, b64_data, mime_type, prompt)
    resp = httpx.post(endpoint, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)

    logger.debug("Response status: %s", resp.status_code)

    RESPONSE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESPONSE_OUTPUT_PATH.write_text(resp.text, encoding="utf-8")

    resp.raise_for_status()
    return _parse_response(resp.json())


def _get_glm_config() -> tuple[str | None, str, str, dict[str, str]]:
    """Read GLM-OCR env vars and return (api_key, endpoint, model, headers).

    ``GLM_API_KEY`` is optional — vLLM does not require authentication by
    default.  When set, it is sent as a Bearer token.
    """
    api_key = os.environ.get("GLM_API_KEY")
    endpoint = os.environ.get("GLM_API_ENDPOINT", DEFAULT_ENDPOINT)
    model = os.environ.get("GLM_MODEL", DEFAULT_MODEL)

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    return api_key, endpoint, model, headers


def _validate_extension(path: Path) -> str:
    """Return lowercased suffix or raise ValueError."""
    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(_SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type '{suffix}'. Supported: {supported}")
    return suffix


# ---------------------------------------------------------------------------
# Simple mode — whole-page OCR
# ---------------------------------------------------------------------------


def call_glm_ocr(
    file_path: str | Path,
    prompt: str = TASK_TEXT,
) -> str:
    """Extract text from a document or image using GLM-OCR (simple mode).

    For PDFs, each page is rendered to a high-resolution image via PyMuPDF
    and sent individually; results are concatenated with page separators.
    """
    _, endpoint, model, headers = _get_glm_config()
    path = Path(file_path)
    suffix = _validate_extension(path)

    if suffix == ".pdf":
        pages = _pdf_pages_to_base64(path)
        logger.info("Sending OCR for %d PDF page(s) to %s (model %s) ...", len(pages), endpoint, model)
        parts: list[str] = []
        for i, (b64_data, mime_type) in enumerate(pages, 1):
            logger.info("Processing page %d/%d ...", i, len(pages))
            text = _send_ocr_request(endpoint, headers, model, b64_data, mime_type, prompt)
            parts.append(text)
        return "\n\n---\n\n".join(parts)

    b64_data, mime_type = _encode_image(path)
    logger.info("Sending OCR request to %s (model %s) ...", endpoint, model)
    return _send_ocr_request(endpoint, headers, model, b64_data, mime_type, prompt)


# ---------------------------------------------------------------------------
# Layout mode — PP-DocLayoutV3 + per-region OCR
# ---------------------------------------------------------------------------


def _detect_layout(page_img: PILImage.Image) -> list[dict]:
    """Run PP-DocLayoutV3 layout detection on a single page image.

    Returns region dicts sorted in reading order (top->bottom, left->right).
    """
    import numpy as np
    from paddleocr import LayoutDetection

    model = LayoutDetection(model_name="PP-DocLayoutV3")
    img_array = np.array(page_img)
    results = model.predict(img_array, batch_size=1, layout_nms=True)

    regions: list[dict] = []
    for res in results:
        res_json = res.json
        inner = res_json.get("res", res_json)
        for box in inner.get("boxes", []):
            regions.append(box)

    regions.sort(
        key=lambda r: (
            r.get("order", 0) or 0,
            (r.get("coordinate") or [0, 0, 0, 0])[1],
            (r.get("coordinate") or [0, 0, 0, 0])[0],
        )
    )
    return regions


def _process_page_layout(
    page_img: PILImage.Image,
    page_num: int,
    endpoint: str,
    model: str,
    headers: dict[str, str],
    max_workers: int = MAX_WORKERS,
) -> str:
    """Process a single page: detect layout -> OCR each region -> assemble."""
    logger.info("Page %d: detecting layout ...", page_num)
    regions = _detect_layout(page_img)

    ocr_regions = [r for r in regions if r["label"] in _OCR_LABELS]
    logger.info("Page %d: %d layout regions, %d for OCR", page_num, len(regions), len(ocr_regions))

    if not ocr_regions:
        return ""

    crops: list[tuple[int, str, str]] = []
    for idx, region in enumerate(ocr_regions):
        x_min, y_min, x_max, y_max = region["coordinate"]
        cropped = page_img.crop((int(x_min), int(y_min), int(x_max), int(y_max)))
        b64 = _pil_to_base64(cropped)
        crops.append((idx, region["label"], b64))

    results: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _send_ocr_request,
                endpoint,
                headers,
                model,
                b64,
                "image/png",
                _task_prompt_for_label(label),
            ): idx
            for idx, label, b64 in crops
        }
        for future in as_completed(futures):
            region_idx = futures[future]
            try:
                results[region_idx] = future.result()
            except Exception:
                logger.exception("Page %d, region %d: OCR failed", page_num, region_idx)
                results[region_idx] = ""

    parts = [results[idx].strip() for idx in sorted(results) if results[idx].strip()]
    return "\n\n".join(parts)


def call_glm_ocr_layout(
    file_path: str | Path,
    dpi: int = PDF_RENDER_DPI,
    max_workers: int = MAX_WORKERS,
) -> str:
    """Extract text using layout-aware OCR (PP-DocLayoutV3 + GLM-OCR).

    Detects document layout regions, crops each one, sends to GLM-OCR with
    region-type-specific task prompts in parallel, and reassembles in reading
    order.
    """
    from PIL import Image

    _, endpoint, model, headers = _get_glm_config()
    path = Path(file_path)
    suffix = _validate_extension(path)

    if suffix == ".pdf":
        pages = _pdf_pages_to_pil(path, dpi=dpi)
    else:
        pages = [Image.open(path).convert("RGB")]

    logger.info("Layout OCR: %d page(s), endpoint %s, model %s", len(pages), endpoint, model)

    all_parts: list[str] = []
    for page_num, page_img in enumerate(pages, start=1):
        page_text = _process_page_layout(page_img, page_num, endpoint, model, headers, max_workers)
        if page_text:
            if len(pages) > 1:
                all_parts.append(f"<!-- Page {page_num} -->\n\n{page_text}")
            else:
                all_parts.append(page_text)

    return "\n\n---\n\n".join(all_parts)


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def analyze_document(
    source: str | Path,
    output_dir: Path = OUTPUT_DIR,
    mode: str = "simple",
    prompt: str = TASK_TEXT,
    dpi: int = PDF_RENDER_DPI,
    max_workers: int = MAX_WORKERS,
) -> Path:
    """Analyze a document with GLM-OCR and export markdown.

    Args:
        source: Path to the input document file.
        output_dir: Directory for the output markdown file.
        mode: ``"simple"`` for whole-page OCR, ``"layout"`` for
            PP-DocLayoutV3 region detection + per-region OCR.
        prompt: Task prompt (simple mode only). Defaults to
            ``"Text Recognition:"``.  Also supports
            ``"Table Recognition:"`` and ``"Formula Recognition:"``.
        dpi: Resolution for PDF rendering.
        max_workers: Parallel workers for layout mode.

    Returns:
        Path to the generated markdown file.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "layout":
        extracted_text = call_glm_ocr_layout(source, dpi=dpi, max_workers=max_workers)
    else:
        extracted_text = call_glm_ocr(source, prompt)

    md_path = output_dir / "doc.md"
    md_path.write_text(extracted_text, encoding="utf-8")

    logger.info("Done. Output in %s", output_dir.resolve())
    return md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and run GLM-OCR extraction."""
    import argparse

    load_dotenv()

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser(
        description="Extract text from a document using GLM-OCR (zai-org/GLM-OCR).",
    )
    parser.add_argument("file", help="Path to the input document (PDF, JPG, PNG, BMP, WebP).")
    parser.add_argument(
        "-m",
        "--mode",
        choices=["simple", "layout"],
        default="simple",
        help="OCR mode: 'simple' sends whole pages; 'layout' uses PP-DocLayoutV3 "
        "for region detection + per-region OCR (default: simple).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory (default: ./output).",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        default=TASK_TEXT,
        help="Task prompt (simple mode only). Supports: "
        "'Text Recognition:', 'Table Recognition:', 'Formula Recognition:' "
        f"(default: '{TASK_TEXT}').",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=PDF_RENDER_DPI,
        help=f"DPI for PDF rendering (default: {PDF_RENDER_DPI}).",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=f"Parallel OCR workers per page, layout mode (default: {MAX_WORKERS}).",
    )
    args = parser.parse_args()

    md_path = analyze_document(
        args.file,
        args.output_dir,
        args.mode,
        args.prompt,
        args.dpi,
        args.workers,
    )
    print(f"Markdown written to {md_path}")


if __name__ == "__main__":
    main()
