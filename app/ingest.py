"""Ingestion command: load PDFs, text files and web pages into the vector store.

Usage:
    python -m app.ingest --customer demo docs/
    python -m app.ingest --customer demo docs/handbook.pdf https://example.com/faq
    python -m app.ingest --customer demo --reset docs/     # wipe this customer first

Running it again on the same file re-indexes that file (no duplicates).
"""
import argparse
import re
import sys
from pathlib import Path

from . import config
from .store import Store

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TEXT_SUFFIXES = {".txt", ".md"}


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, preferring to cut at sentence ends."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
            if cut > size // 2:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_pdf(path: Path) -> list[dict]:
    from pypdf import PdfReader

    out = []
    for page_number, page in enumerate(PdfReader(str(path)).pages, start=1):
        for piece in chunk_text(page.extract_text() or ""):
            out.append({"text": piece, "page": page_number})
    return out


def load_text_file(path: Path) -> list[dict]:
    return [{"text": piece, "page": 0} for piece in chunk_text(path.read_text(encoding="utf-8"))]


def load_url(url: str) -> list[dict]:
    import httpx
    from bs4 import BeautifulSoup

    resp = httpx.get(url, follow_redirects=True, timeout=30, headers={"User-Agent": "doc-assistant/1.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return [{"text": piece, "page": 0} for piece in chunk_text(text)]


def expand_targets(targets: list[str]) -> list[str]:
    out = []
    for target in targets:
        if target.startswith(("http://", "https://")):
            out.append(target)
            continue
        path = Path(target)
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.suffix.lower() in TEXT_SUFFIXES | {".pdf"}:
                    out.append(str(child))
        elif path.exists():
            out.append(str(path))
        else:
            print(f"  skipped (not found): {target}", file=sys.stderr)
    return out


def ingest(store: Store, customer_id: str, targets: list[str]) -> dict[str, int]:
    """Index each target for one customer. Returns {source: chunk_count}."""
    results = {}
    for target in expand_targets(targets):
        if target.startswith(("http://", "https://")):
            source, chunks = target, load_url(target)
        else:
            path = Path(target)
            source = path.name
            chunks = load_pdf(path) if path.suffix.lower() == ".pdf" else load_text_file(path)
        results[source] = store.replace_document(customer_id, source, chunks)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Load documents into the vector store.")
    parser.add_argument("--customer", required=True, help="Customer id these documents belong to")
    parser.add_argument("--reset", action="store_true", help="Delete this customer's index first")
    parser.add_argument("targets", nargs="+", help="Files, folders or URLs")
    args = parser.parse_args()

    store = Store(config.DATA_DIR / "chroma")
    if args.reset:
        store.delete_customer(args.customer)
        print(f"Cleared index for customer '{args.customer}'")
    results = ingest(store, args.customer, args.targets)
    for source, count in results.items():
        print(f"  {source}: {count} chunks")
    print(f"Done. {len(results)} documents, {sum(results.values())} chunks for '{args.customer}'.")


if __name__ == "__main__":
    main()
