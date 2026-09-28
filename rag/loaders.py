"""Load distillery outputs into uniform segments.

A *segment* is the smallest self-contained unit of a document: one transcript
utterance, one OCR page, one plain-text paragraph. Chunking (chunker.py) merges
segments; metadata flows through untouched.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

TIMESTAMP_RE = re.compile(r"^\[(\d{1,3}):(\d{2})\]\s?")
PAGE_HEADER_RE = re.compile(r"^=====\s*第\s*(\d+)\s*页\s*=====$")
# ocr_pdf.py writes this sentinel on pages where nothing was recognized; it
# carries no information, so it is dropped instead of polluting the index.
OCR_EMPTY_PAGE = "(本页未识别到文字)"
# SenseVoice annotates non-speech events with pictographs (😊 🎼 ...); strip them.
NOISE_MARKS_RE = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]")


@dataclass
class Segment:
    text: str
    metadata: dict = field(default_factory=dict)


def _clean(text: str) -> str:
    return NOISE_MARKS_RE.sub("", text).strip()


def load_transcript(path: Path) -> list[Segment]:
    """Parse `distillery transcript.txt`: one `[MM:SS] utterance` per line."""
    segments: list[Segment] = []
    source = path.stem
    for line in path.read_text(encoding="utf-8").splitlines():
        match = TIMESTAMP_RE.match(line)
        if not match:
            continue
        minutes, seconds = int(match.group(1)), int(match.group(2))
        text = _clean(TIMESTAMP_RE.sub("", line, count=1))
        if not text:
            continue
        segments.append(
            Segment(
                text,
                {
                    "source": source,
                    "kind": "transcript",
                    "timestamp": f"{minutes:02d}:{seconds:02d}",
                    "start_seconds": minutes * 60 + seconds,
                },
            )
        )
    return segments


def load_ocr(path: Path) -> list[Segment]:
    """Parse `ocr_pdf.py` output: pages separated by `===== 第 N 页 =====`."""
    segments: list[Segment] = []
    source = path.stem
    page: int | None = None
    buffer: list[str] = []

    def flush() -> None:
        if page is None:
            return
        text = _clean("\n".join(buffer))
        if text and text != OCR_EMPTY_PAGE:
            segments.append(Segment(text, {"source": source, "kind": "ocr", "page": page}))

    for line in path.read_text(encoding="utf-8").splitlines():
        match = PAGE_HEADER_RE.match(line.strip())
        if match:
            flush()
            page = int(match.group(1))
            buffer = []
        elif page is not None:
            buffer.append(line)
    flush()
    return segments


def load_plain(path: Path) -> list[Segment]:
    """Fall back to blank-line paragraphs (e.g. vision.py captions)."""
    source = path.stem
    paragraphs = (
        _clean(block)
        for block in path.read_text(encoding="utf-8").split("\n\n")
    )
    return [
        Segment(text, {"source": source, "kind": "plain"})
        for text in paragraphs
        if text
    ]


def load_any(path: Path) -> tuple[str, list[Segment]]:
    """Sniff the format from the first non-empty lines, then dispatch."""
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return "plain", []
    if any(TIMESTAMP_RE.match(line) for line in lines[:10]):
        return "transcript", load_transcript(path)
    if any(PAGE_HEADER_RE.match(line.strip()) for line in lines[:10]):
        return "ocr", load_ocr(path)
    return "plain", load_plain(path)
