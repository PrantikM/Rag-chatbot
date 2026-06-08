"""
Multimodal PDF processor for the RAG chatbot.

Handles three content types from research papers:
  1. Text + Tables  → extracted as Markdown via pymupdf4llm
  2. Images/Figures → extracted via pymupdf, captioned by Llama 4 Scout
  3. Formulas       → captured as image regions, described by vision model

Returns a unified list of LangChain Documents ready for vector indexing.
"""

import base64
import io
import logging

import pymupdf  # PDF image extraction
import pymupdf4llm  # PDF-to-Markdown conversion
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_IMAGE_WIDTH = 150  # px – skip icons, logos, decorative elements
MIN_IMAGE_HEIGHT = 150
MAX_IMAGE_BYTES = 3_500_000  # stay under Groq's 4 MB base64 limit
IMAGE_CAPTION_PROMPT = (
    "You are analyzing an image extracted from a scientific research paper. "
    "Provide a detailed, accurate description following these rules:\n\n"
    "• If the image contains a **mathematical formula or equation**, transcribe "
    "it precisely using LaTeX notation (e.g., $E = mc^2$). Also explain what "
    "the formula represents in plain language.\n\n"
    "• If the image contains a **graph, chart, or plot**, describe the axes, "
    "the data trends, key data points, and any conclusions that can be drawn.\n\n"
    "• If the image contains a **diagram, flowchart, or architecture**, describe "
    "all components, their labels, and how they connect.\n\n"
    "• If the image contains a **table**, reproduce its content in Markdown "
    "table format.\n\n"
    "Be thorough — your description will be used to answer questions about "
    "this paper, so include every detail visible in the image."
)


# ---------------------------------------------------------------------------
# Text Extraction
# ---------------------------------------------------------------------------
def extract_text_chunks(pdf_path: str) -> list[Document]:
    """
    Extract structured Markdown text from a PDF and split into chunks.

    Uses pymupdf4llm for Markdown conversion (preserves headings, tables,
    lists) and LangChain's RecursiveCharacterTextSplitter for chunking.

    Returns:
        List of LangChain Document objects with page metadata.
    """
    # Get per-page Markdown with metadata
    page_chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""],
    )

    documents = []
    for chunk in page_chunks:
        page_text = chunk.get("text", "")
        page_meta = chunk.get("metadata", {})
        page_num = page_meta.get("page", 0) + 1  # 1-indexed for display

        if not page_text.strip():
            continue

        splits = splitter.split_text(page_text)
        for i, split in enumerate(splits):
            documents.append(
                Document(
                    page_content=split,
                    metadata={
                        "source": pdf_path,
                        "page": page_num,
                        "content_type": "text",
                        "chunk_index": i,
                    },
                )
            )

    logger.info("Extracted %d text chunks from %s", len(documents), pdf_path)
    return documents


# ---------------------------------------------------------------------------
# Image Extraction
# ---------------------------------------------------------------------------
def extract_images(
    pdf_path: str,
    min_width: int = MIN_IMAGE_WIDTH,
    min_height: int = MIN_IMAGE_HEIGHT,
) -> list[tuple[int, bytes, str]]:
    """
    Extract embedded images from a PDF, filtering out small/decorative ones.

    Returns:
        List of (page_number, image_bytes, image_extension) tuples.
        page_number is 1-indexed.
    """
    doc = pymupdf.open(pdf_path)
    images = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)

        for img_info in image_list:
            xref = img_info[0]
            width = img_info[2]
            height = img_info[3]

            # Skip small images (icons, logos, bullets)
            if width < min_width or height < min_height:
                continue

            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Resize if too large for Groq's base64 limit
                image_bytes = _resize_if_needed(image_bytes, image_ext)

                images.append((page_index + 1, image_bytes, image_ext))
            except Exception as e:
                logger.warning(
                    "Failed to extract image xref=%d on page %d: %s",
                    xref,
                    page_index + 1,
                    e,
                )

    doc.close()
    logger.info("Extracted %d images from %s", len(images), pdf_path)
    return images


def _resize_if_needed(image_bytes: bytes, image_ext: str) -> bytes:
    """Resize an image if it exceeds the base64 size limit for Groq."""
    if len(image_bytes) <= MAX_IMAGE_BYTES:
        return image_bytes

    img = Image.open(io.BytesIO(image_bytes))

    # Progressively reduce size until under limit
    quality = 85
    scale = 0.8
    for _ in range(5):  # max 5 attempts
        new_size = (int(img.width * scale), int(img.height * scale))
        resized = img.resize(new_size, Image.LANCZOS)

        buf = io.BytesIO()
        save_format = "JPEG" if image_ext.lower() in ("jpg", "jpeg") else "PNG"
        resized.save(buf, format=save_format, quality=quality)
        result = buf.getvalue()

        if len(result) <= MAX_IMAGE_BYTES:
            return result

        scale *= 0.8
        quality = max(quality - 10, 40)

    # Last resort: return whatever we got
    logger.warning("Image still large after resizing (%d bytes)", len(result))
    return result


# ---------------------------------------------------------------------------
# Image Captioning
# ---------------------------------------------------------------------------
def caption_image(llm, image_bytes: bytes, image_ext: str) -> str:
    """
    Generate a detailed text caption for an image using a vision LLM.

    Sends the image as base64 to the multimodal LLM with a specialized
    prompt for research paper content (graphs, formulas, diagrams).

    Returns:
        Caption string describing the image content.
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Map extension to MIME type
    mime_map = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}
    mime_type = mime_map.get(image_ext.lower(), "image/png")

    message = HumanMessage(
        content=[
            {"type": "text", "text": IMAGE_CAPTION_PROMPT},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64}"},
            },
        ]
    )

    response = llm.invoke([message])
    return response.content


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def process_pdf(pdf_path: str, llm, progress_callback=None) -> list[Document]:
    """
    Full multimodal PDF processing pipeline.

    Steps:
        1. Extract structured text as Markdown → chunk into Documents
        2. Extract embedded images (filtered by size)
        3. Caption each image with the vision LLM
        4. Combine text chunks + image caption Documents

    Args:
        pdf_path:          Path to the PDF file.
        llm:               A multimodal LLM instance (e.g., Llama 4 Scout via Groq).
        progress_callback: Optional callable(stage: str, current: int, total: int)
                           for reporting progress to the UI.

    Returns:
        Combined list of LangChain Document objects ready for embedding.
    """
    all_documents = []

    # --- Stage 1: Text extraction ---
    if progress_callback:
        progress_callback("Extracting text and tables...", 0, 3)

    text_docs = extract_text_chunks(pdf_path)
    all_documents.extend(text_docs)

    # --- Stage 2: Image extraction ---
    if progress_callback:
        progress_callback("Extracting images and figures...", 1, 3)

    images = extract_images(pdf_path)

    # --- Stage 3: Image captioning ---
    if images:
        total_images = len(images)
        for idx, (page_num, img_bytes, img_ext) in enumerate(images):
            if progress_callback:
                progress_callback(
                    f"Analyzing figure {idx + 1}/{total_images}...", 2, 3
                )

            try:
                caption = caption_image(llm, img_bytes, img_ext)
                caption_doc = Document(
                    page_content=(
                        f"[Figure on page {page_num}]\n\n{caption}"
                    ),
                    metadata={
                        "source": pdf_path,
                        "page": page_num,
                        "content_type": "image_caption",
                        "image_index": idx + 1,
                    },
                )
                all_documents.append(caption_doc)
            except Exception as e:
                logger.warning(
                    "Failed to caption image %d on page %d: %s",
                    idx + 1,
                    page_num,
                    e,
                )
    else:
        if progress_callback:
            progress_callback("No figures found to analyze.", 2, 3)

    if progress_callback:
        progress_callback("Processing complete!", 3, 3)

    logger.info(
        "Total documents: %d (%d text, %d image captions)",
        len(all_documents),
        len(text_docs),
        len(all_documents) - len(text_docs),
    )
    return all_documents
