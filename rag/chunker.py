"""Merge adjacent segments into retrieval-sized chunks with sliding overlap.

Greedy packing keeps related utterances together; the last `overlap` segments
of a finished group are carried into the next one so context is not cut
mid-thought. Positional metadata (timestamp / page) of each group is kept as a
start-end range, which is what powers "jump back into the video" citations.
"""
from .config import CHUNK_MAX_CHARS, CHUNK_OVERLAP_SEGMENTS
from .loaders import Segment


def merge_segments(
    segments: list[Segment],
    max_chars: int = CHUNK_MAX_CHARS,
    overlap: int = CHUNK_OVERLAP_SEGMENTS,
) -> list[Segment]:
    if not segments:
        return []

    groups: list[list[Segment]] = []
    current: list[Segment] = []
    size = 0
    for seg in segments:
        added = len(seg.text)
        if current and size + added > max_chars:
            groups.append(current)
            current = current[-overlap:] if overlap > 0 else []
            size = sum(len(s.text) for s in current)
        current.append(seg)
        size += added
    if current:
        groups.append(current)

    chunks: list[Segment] = []
    for group in groups:
        metadata = dict(group[0].metadata)
        for key in ("timestamp", "page"):
            first, last = group[0].metadata.get(key), group[-1].metadata.get(key)
            if first is None:
                continue
            metadata[key] = first if first == last else f"{first}-{last}"
        chunks.append(Segment("\n".join(seg.text for seg in group), metadata))
    return chunks
