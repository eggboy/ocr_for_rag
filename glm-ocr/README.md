# GLM OCR

An OCR tool powered by [**GLM-OCR**](https://huggingface.co/zai-org/GLM-OCR) — a dedicated 0.9B multimodal OCR model by [Z.ai](https://zai.org/), ranked #1 on OmniDocBench V1.5 (94.62). The model is self-hosted via [vLLM](https://docs.vllm.ai/) and accessed through an OpenAI-compatible API.

## Quick Start

**1. Start the model server** (requires a GPU):

```bash
# Option A: Docker
docker build -f Dockerfile.vllm -t glm-ocr-vllm .
docker run --gpus all -p 8000:8000 glm-ocr-vllm

# Option B: vLLM directly
vllm serve zai-org/GLM-OCR --host 0.0.0.0 --port 8000 --trust-remote-code
```

**2. Install the CLI**:

```bash
uv sync
```

**3. Run OCR**:

```bash
# Extract text from a PDF
uv run glm-ocr document.pdf

# Extract tables
uv run glm-ocr invoice.png -p "Table Recognition:"

# Mixed document with auto-detected regions (requires: uv sync --extra layout)
uv run glm-ocr paper.pdf --mode layout
```

Output is written to `./output/doc.md`.

## Overview

This tool offers two OCR modes:

| Mode | How it works | When to use |
|---|---|---|
| **simple** (default) | Renders each PDF page (or takes the image as-is) and sends the **whole page** to GLM-OCR with a task prompt (`Text Recognition:`, `Table Recognition:`, or `Formula Recognition:`). | Most documents — the model handles mixed layouts well on its own. Use `Text Recognition:` as the default; switch to `Table Recognition:` or `Formula Recognition:` only when the entire input is that content type. |
| **layout** | Runs **PP-DocLayoutV3** locally to detect document regions (text, tables, formulas, etc.), crops each region, and **automatically selects the right task prompt** per region type. Results are sent to GLM-OCR in parallel and reassembled in reading order. | Mixed documents with tables, formulas, and text on the same page. **No manual prompt selection needed** — the layout detector picks the right prompt for each region. |

Both modes use [PyMuPDF](https://pymupdf.readthedocs.io/) to render PDF pages to high-resolution images — no poppler or external tools required.

> **Which mode should I use?** For most real-world documents (mixed text, tables, and formulas on the same page), use `--mode layout`. It removes the need to inspect each page and choose a prompt — PP-DocLayoutV3 classifies regions automatically and routes each one to the optimal task prompt. Use simple mode when you know the entire input is a single content type, or when you don't need the layout dependency.

## Model

| | |
|---|---|
| **Model** | [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) |
| **Parameters** | 0.9B (CogViT visual encoder + GLM-0.5B language decoder) |
| **Benchmark** | #1 on OmniDocBench V1.5 (94.62) |
| **Serving** | vLLM, SGLang, or Ollama |

The model natively supports three task prompts:

### Text Recognition

General-purpose OCR — extracts all visible text and preserves structure as Markdown.

```bash
uv run glm-ocr scanned-report.pdf -p "Text Recognition:"
```

```python
text = call_glm_ocr("scanned-report.pdf", prompt="Text Recognition:")
```

Example output:

```markdown
# Quarterly Revenue Report

Revenue for Q3 2025 reached **$4.2B**, a 12% increase year-over-year.

## Regional Breakdown

- North America: $2.1B
- EMEA: $1.3B
- APAC: $0.8B
```

### Table Recognition

Extracts tabular data into structured Markdown tables.

```bash
uv run glm-ocr financial-statement.png -p "Table Recognition:"
```

```python
text = call_glm_ocr("financial-statement.png", prompt="Table Recognition:")
```

Example output:

```markdown
| Item | Q2 2025 | Q3 2025 | Change |
|---|---|---|---|
| Revenue | $3.8B | $4.2B | +10.5% |
| Operating Income | $1.1B | $1.3B | +18.2% |
| Net Income | $0.9B | $1.1B | +22.2% |
```

### Formula Recognition

Extracts mathematical equations and formulas as LaTeX.

```bash
uv run glm-ocr equation-sheet.png -p "Formula Recognition:"
```

```python
text = call_glm_ocr("equation-sheet.png", prompt="Formula Recognition:")
```

Example output:

```latex
E = mc^2

\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u

\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
```

## Requirements

- Python 3.11+
- A running vLLM instance serving `zai-org/GLM-OCR` (see [Deployment](#deployment))

## Installation

```bash
# Simple mode only (lightweight — just httpx + pymupdf)
uv sync

# With layout mode support (adds paddleocr, numpy, pillow)
uv sync --extra layout
```

## Deployment

### vLLM (recommended)

Use the included Dockerfile:

```bash
docker build -f Dockerfile.vllm -t glm-ocr-vllm .
docker run --gpus all -p 8000:8000 glm-ocr-vllm
```

Or run vLLM directly:

```bash
vllm serve zai-org/GLM-OCR --host 0.0.0.0 --port 8000 --trust-remote-code
```

The model will be available at `http://localhost:8000/v1/chat/completions`.

### Official SDK

A separate Dockerfile uses the official `glmocr` SDK with `[server,selfhosted]` extras:

```bash
docker build -f Dockerfile -t glm-ocr-sdk .
```

## Configuration

Set the following environment variables (or use a `.env` file):

| Variable | Required | Default | Description |
|---|---|---|---|
| `GLM_API_KEY` | No | — | Bearer token for the vLLM endpoint (only needed if auth is enabled) |
| `GLM_API_ENDPOINT` | No | `http://localhost:8000/v1/chat/completions` | OpenAI-compatible API endpoint URL |
| `GLM_MODEL` | No | `zai-org/GLM-OCR` | Model identifier |

## Usage

### CLI

```bash
# Simple mode — send whole pages to the model
uv run glm-ocr document.pdf
uv run glm-ocr photo.jpg -o ./results

# Layout mode — detect regions first, then OCR each region
uv run glm-ocr document.pdf --mode layout
uv run glm-ocr document.pdf --mode layout --dpi 300 --workers 4

# Use a specific task prompt
uv run glm-ocr invoice.png -p "Table Recognition:"
uv run glm-ocr equation.png -p "Formula Recognition:"
```

### CLI options

```
positional arguments:
  file                  Path to the input document (PDF, JPG, PNG, BMP, WebP)

options:
  -m, --mode {simple,layout}
                        OCR mode (default: simple)
  -o, --output-dir PATH
                        Output directory (default: ./output)
  -p, --prompt TEXT     Task prompt (default: "Text Recognition:")
                        Also supports: "Table Recognition:", "Formula Recognition:"
  --dpi INT             DPI for PDF rendering (default: 300)
  -w, --workers INT     Parallel OCR workers per page, layout mode (default: 4)
```

### Python API

```python
from glm_ocr import call_glm_ocr, call_glm_ocr_layout, analyze_document

# Simple mode (default: "Text Recognition:")
text = call_glm_ocr("document.pdf")

# Table extraction
text = call_glm_ocr("table.png", prompt="Table Recognition:")

# Layout mode — auto-selects the right task prompt per region type
text = call_glm_ocr_layout("document.pdf", dpi=300, max_workers=4)

# Either mode, with markdown file output
md_path = analyze_document("document.pdf", mode="simple")
md_path = analyze_document("document.pdf", mode="layout")
```

## How it works

### Simple mode

```
PDF/Image → PyMuPDF (render pages) → base64 encode → GLM-OCR API → Markdown
```

Each page is rendered at 300 DPI (configurable via `--dpi`) and sent as a single base64-encoded image with a task prompt.

For multi-page PDFs, results are joined with `---` separators.

### Layout mode

```
PDF/Image → PyMuPDF (render pages) → PP-DocLayoutV3 (detect regions)
         → crop each region → GLM-OCR API (parallel, per-region task prompts)
         → reassemble in reading order → Markdown
```

[PP-DocLayoutV3](https://github.com/PaddlePaddle/PaddleOCR) is a document layout analysis model (~132 MB) that runs **locally** — it does not call any external API. It detects structural regions like text blocks, tables, formulas, headers, and figures.

In layout mode, each detected region is sent to GLM-OCR with the most appropriate task prompt:

| Region label | Task prompt |
|---|---|
| `table` | `Table Recognition:` |
| `formula` | `Formula Recognition:` |
| All others (text, title, header, ...) | `Text Recognition:` |

Regions are processed in parallel (configurable with `--workers`) and reassembled in the original reading order.

## Supported file types

- **Images:** JPG, JPEG, PNG, BMP, WebP
- **Documents:** PDF (multi-page supported)
