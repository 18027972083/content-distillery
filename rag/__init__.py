"""content-distillery RAG module (v0.2).

Pipeline: loaders -> chunker -> embeddings -> store -> ask.

Documents are the distillery's own outputs (video transcripts, OCR text,
image captions), so retrieval results carry positional metadata: a timestamp
to jump back into the source video, or a page number into the source PDF.
"""
