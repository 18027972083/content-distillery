# -*- coding: utf-8 -*-
"""扫描版 PDF 的 OCR 工具：PyMuPDF 渲染页面 → RapidOCR 识别 → 输出文本文件
用法:
  python ocr_pdf.py <pdf路径> [输出txt路径] [--dpi 300] [--pages 1-20]
示例:
  python ocr_pdf.py "D:/books/scanned.pdf"
  python ocr_pdf.py "D:/books/scanned.pdf" -o out.txt --pages 1-10 --dpi 240
"""
import argparse
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
from rapidocr_onnxruntime import RapidOCR


def parse_pages(spec: str, total: int):
    """解析 '1-20' / '3' / '5,10-12' 形式的页码指定，返回 0-based 页面索引列表"""
    if not spec:
        return list(range(total))
    pages = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            start, end = int(a) - 1, int(b) - 1
            pages.update(range(max(0, start), min(total, end + 1)))
        else:
            n = int(part) - 1
            if 0 <= n < total:
                pages.add(n)
    return sorted(pages)


def main():
    ap = argparse.ArgumentParser(description="扫描版 PDF OCR")
    ap.add_argument("pdf", help="PDF 文件路径")
    ap.add_argument("-o", "--output", default=None, help="输出 txt 路径（默认同目录同名 .txt）")
    ap.add_argument("--dpi", type=int, default=300, help="渲染分辨率（默认 300）")
    ap.add_argument("--pages", default=None, help="页码范围，如 1-20 或 5,10-12")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"文件不存在: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output) if args.output else pdf_path.with_suffix(".ocr.txt")
    ocr = RapidOCR()

    doc = fitz.open(pdf_path)
    pages = parse_pages(args.pages, len(doc))
    print(f"共 {len(doc)} 页，本次处理 {len(pages)} 页 → {out_path}")

    zoom = args.dpi / 72.0
    all_text = []
    t0 = time.time()

    for i, idx in enumerate(pages, 1):
        page = doc[idx]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        img_bytes = pix.tobytes("png")
        result, _ = ocr(img_bytes)
        text = "\n".join(line[1] for line in result) if result else "(本页未识别到文字)"
        all_text.append(f"===== 第 {idx + 1} 页 =====\n{text}\n")
        elapsed = time.time() - t0
        print(f"[{i}/{len(pages)}] 第 {idx+1} 页 完成 ({elapsed:.0f}s)")

    doc.close()
    out_path.write_text("\n".join(all_text), encoding="utf-8")
    print(f"完成！输出: {out_path}（用时 {time.time()-t0:.0f}s）")


if __name__ == "__main__":
    main()
