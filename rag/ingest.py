"""CLI: ingest text files produced by the distillery into the vector store.

Usage:
    python -m rag.ingest transcript.txt notes.txt [more.txt ...] [--db-dir DIR]

Format is sniffed per file ([MM:SS] lines -> transcript, `===== 第 N 页 =====`
-> ocr, otherwise plain paragraphs).
"""
import argparse
import sys
from pathlib import Path

from .chunker import merge_segments
from .config import DEFAULT_DB_DIR
from .embeddings import embed_texts
from .loaders import load_any
from .store import get_store


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Ingest distillery text outputs into the RAG store.")
    parser.add_argument("files", nargs="+", type=Path, help="text files to ingest (transcript / ocr / plain)")
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR, help="vector store directory")
    args = parser.parse_args()

    store = get_store(args.db_dir)
    total_chunks = 0
    for path in args.files:
        kind, segments = load_any(path)
        if not segments:
            print(f"skip {path.name}: no segments parsed ({kind})")
            continue
        chunks = merge_segments(segments)
        vectors = embed_texts([chunk.text for chunk in chunks])
        store.add(chunks, vectors)
        total_chunks += len(chunks)
        print(f"ingested {path.name}: kind={kind}, segments={len(segments)}, chunks={len(chunks)}")

    print(f"done. store now holds {store.count()} chunks in {args.db_dir}")


if __name__ == "__main__":
    main()
