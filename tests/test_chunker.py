from rag.chunker import merge_segments
from rag.loaders import Segment


def test_empty_input():
    assert merge_segments([]) == []


def test_single_segment_passes_metadata_through():
    chunks = merge_segments([Segment("hello", {"source": "a", "kind": "plain"})])
    assert len(chunks) == 1
    assert chunks[0].text == "hello"
    assert chunks[0].metadata["source"] == "a"


def test_segments_pack_up_to_max_chars():
    segments = [Segment("x" * 100, {"source": "s"}) for _ in range(10)]
    chunks = merge_segments(segments, max_chars=250, overlap=0)
    assert all(len(chunk.text) <= 250 for chunk in chunks)
    # join adds one newline between packed segments; strip it when accounting
    assert sum(len(chunk.text.replace("\n", "")) for chunk in chunks) == 1000
    # each group fits 2 segments (201 chars incl. newline), so 10 segments -> 5 chunks
    assert len(chunks) == 5


def test_overlap_carries_tail_into_next_chunk():
    segments = [Segment(str(i) * 50, {"source": "s"}) for i in range(4)]
    chunks = merge_segments(segments, max_chars=100, overlap=1)
    assert len(chunks) >= 2
    # the last segment of a finished group re-opens the next one
    assert chunks[1].text.startswith(segments[1].text)


def test_transcript_timestamp_becomes_range():
    segments = [
        Segment("a" * 80, {"source": "s", "kind": "transcript", "timestamp": "05:00", "start_seconds": 300}),
        Segment("b" * 80, {"source": "s", "kind": "transcript", "timestamp": "05:05", "start_seconds": 305}),
        Segment("c" * 80, {"source": "s", "kind": "transcript", "timestamp": "10:00", "start_seconds": 600}),
    ]
    chunks = merge_segments(segments, max_chars=200, overlap=1)
    single = next(c for c in chunks if c.metadata["timestamp"] == "05:00-05:05")
    assert single.metadata["start_seconds"] == 300


def test_plain_metadata_without_timestamp_is_untouched():
    segments = [Segment("text", {"source": "s", "kind": "plain"}) for _ in range(3)]
    chunks = merge_segments(segments, max_chars=10)
    assert all(chunk.metadata == {"source": "s", "kind": "plain"} for chunk in chunks)
