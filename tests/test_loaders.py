from rag.loaders import load_any


def test_sniffs_and_parses_transcript(tmp_path):
    doc = tmp_path / "lecture.txt"
    doc.write_text("[00:00] Hi everyone. 😊\n[05:00] Second segment here.\n", encoding="utf-8")
    kind, segments = load_any(doc)
    assert kind == "transcript"
    assert len(segments) == 2
    assert segments[0].text == "Hi everyone."  # SenseVoice noise marker stripped
    assert segments[1].metadata["timestamp"] == "05:00"
    assert segments[1].metadata["start_seconds"] == 300


def test_three_digit_minutes_are_parsed(tmp_path):
    doc = tmp_path / "long.txt"
    doc.write_text("[99:00] still going\n[115:00] near the end\n", encoding="utf-8")
    kind, segments = load_any(doc)
    assert kind == "transcript"
    assert len(segments) == 2
    assert segments[1].metadata["start_seconds"] == 115 * 60


def test_sniffs_and_parses_ocr_pages(tmp_path):
    doc = tmp_path / "book.ocr.txt"
    doc.write_text("===== 第 1 页 =====\nfirst page\n\n===== 第 2 页 =====\n(本页未识别到文字)\n", encoding="utf-8")
    kind, segments = load_any(doc)
    assert kind == "ocr"
    assert len(segments) == 1  # empty page is dropped
    assert segments[0].metadata["page"] == 1
    assert segments[0].text == "first page"


def test_falls_back_to_plain_paragraphs(tmp_path):
    doc = tmp_path / "caption.txt"
    doc.write_text("first paragraph\n\nsecond paragraph\n", encoding="utf-8")
    kind, segments = load_any(doc)
    assert kind == "plain"
    assert [segment.text for segment in segments] == ["first paragraph", "second paragraph"]


def test_empty_file_yields_no_segments(tmp_path):
    doc = tmp_path / "empty.txt"
    doc.write_text("", encoding="utf-8")
    kind, segments = load_any(doc)
    assert kind == "plain"
    assert segments == []
