import base64
import logging
import os
from pathlib import Path

import httpx
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"
DEFAULT_MODEL = "mistral-document-ai-2505"
REQUEST_TIMEOUT = 300.0
RESPONSE_OUTPUT_PATH = Path("output/response.json")

DEFAULT_BBOX_SCHEMA: dict = {
    "properties": {
        "document_type": {"title": "Document_Type", "description": "The type of the image.", "type": "string"},
        "short_description": {
            "title": "Short_Description",
            "description": "A description in English describing the image.",
            "type": "string",
        },
        "summary": {"title": "Summary", "description": "Summarize the image.", "type": "string"},
    },
    "required": ["document_type", "short_description", "summary"],
    "title": "BBOXAnnotation",
    "type": "object",
    "additionalProperties": False,
}

_MIME_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}


def _encode_file(file_path: Path) -> tuple[str, str]:
    """Read a file, base64-encode it, and return ``(data_url, url_type)``.

    Args:
        file_path: Path to the input document.

    Returns:
        A tuple of the data-URL string and the URL type key
        (``"image_url"`` or ``"document_url"``).
    """
    b64 = base64.b64encode(file_path.read_bytes()).decode("utf-8")
    suffix = file_path.suffix.lower()
    mime = _MIME_TYPES.get(suffix, "application/pdf")
    url_type = "image_url" if mime.startswith("image") else "document_url"
    data_url = f"data:{mime};base64,{b64}"
    return data_url, url_type


def _build_payload(
    model: str,
    data_url: str,
    url_type: str,
    bbox_schema: dict,
) -> dict:
    """Construct the JSON payload for the Mistral Document AI API.

    Args:
        model: The model identifier to use.
        data_url: Base64-encoded data URL of the document.
        url_type: Either ``"image_url"`` or ``"document_url"``.
        bbox_schema: JSON schema for bbox annotation.

    Returns:
        The request payload dictionary.
    """
    return {
        "model": model,
        "document": {"type": url_type, url_type: data_url},
        "include_image_base64": False,
        "bbox_annotation_format": {
            "type": "json_schema",
            "json_schema": {"schema": bbox_schema, "name": "document_annotation", "strict": True},
        },
    }


def _parse_response(result: dict) -> str:
    """Extract text from the various response formats returned by the API.

    Args:
        result: Parsed JSON response body.

    Returns:
        Extracted markdown/text content.

    Raises:
        ValueError: If the response format is unrecognised.
    """
    if pages := result.get("pages"):
        parts: list[str] = []
        for page in pages:
            if not isinstance(page, dict):
                continue
            md = page.get("markdown", "")
            for img in page.get("images", []):
                ann = img.get("image_annotation")
                if ann:
                    img_id = img.get("id", "unknown")
                    ann_text = _format_annotation(ann)
                    md = md.replace(
                        f"![{img_id}]({img_id})",
                        f"![{img_id}]({img_id})\n\n> **[{img_id}]:** {ann_text}",
                    )
            parts.append(md)
        return "\n\n".join(parts)

    if "content" in result:
        return result["content"]
    if "text" in result:
        return result["text"]
    if choices := result.get("choices"):
        return choices[0].get("message", {}).get("content", "")

    logger.warning("Unrecognised response format — returning empty string. Keys: %s", list(result.keys()))
    return ""


def _format_annotation(ann: dict | str) -> str:
    """Return a human-readable string for an image annotation.

    Args:
        ann: Annotation value — either a dict with ``short_description``/``summary`` or a plain string.

    Returns:
        Formatted annotation text.
    """
    if isinstance(ann, dict):
        desc = ann.get("short_description", "")
        summary = ann.get("summary", "")
        return f"{desc}\n{summary}" if summary else desc
    return str(ann)


def call_mistral_doc_ai(file_path: str | Path, json_schema: dict | None = None) -> str:
    """Extract text from a document using Mistral Document AI.

    Args:
        file_path: Path to the input document (PDF, JPG, PNG).
        json_schema: Optional JSON schema for bbox annotation format.
            Falls back to :data:`DEFAULT_BBOX_SCHEMA` when *None*.

    Returns:
        Extracted markdown text from the document.

    Raises:
        ValueError: If ``MISTRAL_DOC_AI_ENDPOINT`` is not set.
        httpx.HTTPStatusError: If the API returns a non-2xx response.
    """
    credential = DefaultAzureCredential()
    token = credential.get_token(COGNITIVE_SERVICES_SCOPE)

    endpoint = os.environ.get("MISTRAL_DOC_AI_ENDPOINT")
    model = os.environ.get("MISTRAL_DOC_AI_MODEL", DEFAULT_MODEL)
    if not endpoint:
        raise ValueError("MISTRAL_DOC_AI_ENDPOINT must be set in .env")

    path = Path(file_path)
    data_url, url_type = _encode_file(path)

    bbox_schema = json_schema or DEFAULT_BBOX_SCHEMA
    payload = _build_payload(model, data_url, url_type, bbox_schema)

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token.token}"}
    resp = httpx.post(endpoint, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)

    logger.debug("Response status: %s", resp.status_code)
    logger.debug("Response headers: %s", dict(resp.headers))

    RESPONSE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESPONSE_OUTPUT_PATH.write_text(resp.text, encoding="utf-8")
    logger.debug("Response body written to %s", RESPONSE_OUTPUT_PATH)

    resp.raise_for_status()
    return _parse_response(resp.json())


def main() -> None:
    """Parse CLI arguments and run document extraction."""
    import argparse

    load_dotenv()

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser(description="Extract text from a document using Mistral Document AI.")
    parser.add_argument("file", help="Path to the input document (PDF, JPG, PNG).")
    args = parser.parse_args()

    extracted_text = call_mistral_doc_ai(args.file)
    print(extracted_text)


if __name__ == "__main__":
    main()
