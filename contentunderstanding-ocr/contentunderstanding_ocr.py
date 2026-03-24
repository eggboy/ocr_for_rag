import logging
import os
from pathlib import Path

from azure.ai.contentunderstanding import ContentUnderstandingClient
from azure.ai.contentunderstanding.models import (
    AnalyzeResult,
    DocumentContent,
)
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("./output")


def _build_client() -> ContentUnderstandingClient:
    """Create an authenticated ``ContentUnderstandingClient``.

    Uses ``CONTENTUNDERSTANDING_ENDPOINT`` (required) and
    ``CONTENTUNDERSTANDING_KEY`` (optional – falls back to
    ``DefaultAzureCredential``).

    Returns:
        A ready-to-use Content Understanding client.
    """
    endpoint = os.environ["CONTENTUNDERSTANDING_ENDPOINT"]
    key = os.getenv("CONTENTUNDERSTANDING_KEY")
    credential = AzureKeyCredential(key) if key else DefaultAzureCredential()
    return ContentUnderstandingClient(endpoint=endpoint, credential=credential)


def analyze_document(
    source: str | Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Analyze a document with Azure Content Understanding and export markdown.

    Args:
        source: Path to the input document file.
        output_dir: Directory for the output markdown file.

    Returns:
        Path to the generated markdown file.

    Raises:
        FileNotFoundError: If *source* does not exist.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")

    output_dir.mkdir(parents=True, exist_ok=True)

    client = _build_client()

    with open(source, "rb") as f:
        file_bytes = f.read()

    logger.info("Analyzing %s with prebuilt-documentSearch...", source)
    poller = client.begin_analyze_binary(
        analyzer_id="prebuilt-documentSearch",
        binary_input=file_bytes,
    )
    result: AnalyzeResult = poller.result()

    # A PDF file has only one content element even if it contains multiple pages
    content = result.contents[0]

    # Log document properties
    if isinstance(content, DocumentContent):
        logger.info("Document type: %s", content.mime_type or "(unknown)")
        logger.info("Pages: %d–%d", content.start_page_number, content.end_page_number)

        if content.pages:
            unit = content.unit or "units"
            for page in content.pages:
                logger.info(
                    "  Page %d: %s x %s %s",
                    page.page_number,
                    page.width,
                    page.height,
                    unit,
                )

        if content.tables:
            logger.info("Number of tables: %d", len(content.tables))
            for i, table in enumerate(content.tables, 1):
                logger.info(
                    "  Table %d: %d rows x %d columns",
                    i,
                    table.row_count,
                    table.column_count,
                )

    md_path = output_dir / "doc.md"
    md_path.write_text(content.markdown, encoding="utf-8")

    logger.info("Done. Output in %s", output_dir.resolve())
    return md_path


def main() -> None:
    """Parse CLI arguments and run document analysis."""
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Analyze a document using Azure Content Understanding.",
    )
    parser.add_argument("file", help="Path to the input document file.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: ./output).",
    )
    args = parser.parse_args()

    md_path = analyze_document(args.file, args.output_dir)
    print(f"Markdown written to {md_path}")


if __name__ == "__main__":
    main()
