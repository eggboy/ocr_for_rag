import logging
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode

logger = logging.getLogger(__name__)

MIN_WIDTH = 100
MIN_HEIGHT = 100
SKIP_CLASSES = frozenset({"logo", "signature", "icon", "photograph"})
DEFAULT_OUTPUT_DIR = Path("./output")
IMAGE_PLACEHOLDER_PATTERNS = ("<!-- image -->\n\n", "<!-- image -->")


def _build_converter() -> DocumentConverter:
    """Create a ``DocumentConverter`` configured for PDF extraction.

    Returns:
        A ready-to-use document converter with picture classification
        and table-structure extraction enabled.
    """
    pipeline_options = PdfPipelineOptions(
        generate_picture_images=True,
        images_scale=2.0,
        do_picture_classification=True,
        do_table_structure=True,
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def _filter_pictures(doc: object) -> None:
    """Remove unwanted pictures from a Docling document **in-place**.

    Pictures that are too small (below :data:`MIN_WIDTH` / :data:`MIN_HEIGHT`)
    or classified as a skippable type (see :data:`SKIP_CLASSES`) have their
    image data set to ``None`` so they won't appear in the exported markdown.

    Args:
        doc: A Docling ``DoclingDocument`` instance.
    """
    for i, pic in enumerate(doc.pictures):
        reason = _should_skip_picture(pic)
        if reason:
            logger.info("Filtering picture %d: %s", i, reason)
            pic.image = None


def _should_skip_picture(pic: object) -> str:
    """Return a reason string if *pic* should be filtered, or empty string to keep it.

    Args:
        pic: A single picture element from a Docling document.

    Returns:
        A non-empty reason string if the picture should be removed,
        or ``""`` if it should be kept.
    """
    if pic.prov:
        bbox = pic.prov[0].bbox
        if bbox.width < MIN_WIDTH or bbox.height < MIN_HEIGHT:
            return f"too small ({bbox.width:.0f}x{bbox.height:.0f})"

    if pic.meta and pic.meta.classification:
        main_pred = pic.meta.classification.get_main_prediction()
        if main_pred and main_pred.class_name in SKIP_CLASSES:
            return f"classified as {main_pred.class_name}"

    return ""


def _strip_image_placeholders(md_path: Path) -> None:
    """Remove leftover ``<!-- image -->`` placeholders from a markdown file.

    Args:
        md_path: Path to the markdown file to clean up.
    """
    text = md_path.read_text(encoding="utf-8")
    for pattern in IMAGE_PLACEHOLDER_PATTERNS:
        text = text.replace(pattern, "")
    md_path.write_text(text, encoding="utf-8")


def convert_pdf_to_markdown(
    source: str | Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Convert a PDF document to filtered markdown with referenced images.

    Unwanted pictures (icons, logos, signatures, small images) are removed
    before export so the resulting markdown only references meaningful visuals.

    Args:
        source: Path to the input PDF file.
        output_dir: Directory for the output markdown and images.

    Returns:
        Path to the generated markdown file.

    Raises:
        FileNotFoundError: If *source* does not exist.
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")

    output_dir.mkdir(parents=True, exist_ok=True)

    converter = _build_converter()
    result = converter.convert(str(source))
    doc = result.document

    _filter_pictures(doc)

    md_path = output_dir / "doc.md"
    doc.save_as_markdown(md_path, image_mode=ImageRefMode.REFERENCED)
    _strip_image_placeholders(md_path)

    logger.info("Done. Output in %s", output_dir.resolve())
    return md_path


def main() -> None:
    """Parse CLI arguments and run PDF-to-markdown conversion."""
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Convert a PDF to markdown using Docling with picture filtering.",
    )
    parser.add_argument("file", help="Path to the input PDF document.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: ./output).",
    )
    args = parser.parse_args()

    md_path = convert_pdf_to_markdown(args.file, args.output_dir)
    print(f"Markdown written to {md_path}")


if __name__ == "__main__":
    main()
