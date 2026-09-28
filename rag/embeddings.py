"""Text embeddings via SiliconFlow BAAI/bge-m3 (OpenAI-compatible endpoint).

Kept on urllib on purpose: the distillery's zero-heavy-dependency style.
Batches requests and retries with backoff, because the free tier throttles.
"""
import json
import time
import urllib.error
import urllib.request

from .config import EMBED_BATCH, HTTP_TIMEOUT, SILICONFLOW_EMBED_MODEL, SILICONFLOW_EMBED_URL, require_key


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts; order of the output matches the input."""
    key = require_key("SILICONFLOW_API_KEY")
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start : start + EMBED_BATCH]
        vectors.extend(_post(key, batch))
    return vectors


def _post(key: str, batch: list[str]) -> list[list[float]]:
    payload = json.dumps({"model": SILICONFLOW_EMBED_MODEL, "input": batch}).encode("utf-8")
    request = urllib.request.Request(
        SILICONFLOW_EMBED_URL,
        data=payload,
        method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as response:
                data = json.loads(response.read().decode("utf-8"))
            return [item["embedding"] for item in data["data"]]
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            last_error = RuntimeError(f"embedding request failed: HTTP {exc.code}: {detail}")
            if exc.code not in (429, 500, 502, 503):
                raise last_error from exc
            time.sleep(2**attempt * 2)
    raise last_error  # type: ignore[misc]
