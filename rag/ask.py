"""CLI: ask a question over the ingested knowledge base.

Usage:
    python -m rag.ask "block_size 是什么意思" [--top-k 4] [--db-dir DIR]

Retrieves top-k chunks, has glm-4-flash answer strictly from them, and prints
citations with positional metadata (video timestamp / PDF page) so the answer
can be traced back to the source material.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from .config import (
    CHAT_MAX_TOKENS,
    CHAT_TEMPERATURE,
    DEFAULT_DB_DIR,
    HTTP_TIMEOUT,
    TOP_K,
    ZHIPU_CHAT_MODEL,
    ZHIPU_CHAT_URL,
    require_key,
)
from .embeddings import embed_texts
from .store import get_store

SYSTEM_PROMPT = (
    "你是个人知识库的问答助手。只依据提供的资料片段回答问题；资料不足以回答时明确说明。"
    "引用资料时在句末标注来源编号，如 [1]。回答使用简体中文。"
)


def format_where(metadata: dict) -> str:
    kind = metadata.get("kind")
    if kind == "transcript":
        return f"@ {metadata.get('timestamp', '?')}"
    if kind == "ocr":
        return f"第 {metadata.get('page', '?')} 页"
    return ""


def build_context(hits: list[dict]) -> str:
    blocks = []
    for index, hit in enumerate(hits, start=1):
        where = format_where(hit["metadata"])
        origin = f"{hit['metadata'].get('source', '?')} {where}".strip()
        blocks.append(f"[{index}] ({origin})\n{hit['document']}")
    return "\n\n".join(blocks)


def chat(question: str, context: str) -> str:
    key = require_key("ZHIPU_API_KEY")
    payload = json.dumps(
        {
            "model": ZHIPU_CHAT_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"资料：\n\n{context}\n\n问题：{question}"},
            ],
            "temperature": CHAT_TEMPERATURE,
            "max_tokens": CHAT_MAX_TOKENS,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        ZHIPU_CHAT_URL,
        data=payload,
        method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"chat request failed: HTTP {exc.code}: {detail}") from exc
    return data["choices"][0]["message"]["content"].strip()


def ask(question: str, top_k: int = TOP_K, db_dir: Path = DEFAULT_DB_DIR) -> tuple[str, list[dict]]:
    store = get_store(db_dir)
    if store.count() == 0:
        raise SystemExit("knowledge base is empty; ingest a document first: python -m rag.ingest <file>")
    hits = store.query(embed_texts([question])[0], top_k)
    return chat(question, build_context(hits)), hits


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Ask the distillery knowledge base a question.")
    parser.add_argument("question", help="what to ask")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="number of chunks to retrieve")
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR, help="vector store directory")
    args = parser.parse_args()

    answer, hits = ask(args.question, args.top_k, args.db_dir)
    print(answer)
    print("\n--- 来源 ---")
    for index, hit in enumerate(hits, start=1):
        metadata = hit["metadata"]
        where = format_where(metadata)
        preview = hit["document"][:80].replace("\n", " ")
        print(f"[{index}] {metadata.get('source', '?')} {where} (distance={hit['distance']:.3f}) {preview}...")


if __name__ == "__main__":
    main()
