# -*- coding: utf-8 -*-
"""视觉代理：图片 → 视觉模型 → 文字描述（给纯文本模型"看图"用）
支持任意 OpenAI 兼容端点。
用法:
  python vision.py <图片路径> [更多图片...] [-p "想让它看什么"]
示例:
  # 智谱（默认，永久免费视觉模型）
  python vision.py "D:/Pictures/a.jpg" -p "描述这张图的内容"
  # SiliconFlow（需先有余额）
  python vision.py "D:/Pictures/a.jpg" --base-url https://api.siliconflow.cn/v1/chat/completions \
      --max-tokens 1024
环境变量: ZHIPU_API_KEY 或 SILICONFLOW_API_KEY（也可用 --api-key 传入）
"""
import argparse
import base64
import json
import mimetypes
import os
import sys
import urllib.request
from pathlib import Path

# 强制 UTF-8 输出（Windows 默认 GBK 控制台会乱码）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 默认走智谱（永久免费视觉模型）；SiliconFlow 用 --base-url 显式指定
DEFAULT_MODEL = "glm-4v-flash"
DEFAULT_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
SILICONFLOW_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
SILICONFLOW_URL = "https://api.siliconflow.cn/v1/chat/completions"


def image_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def main():
    ap = argparse.ArgumentParser(description="图片 → 视觉模型 → 文字")
    ap.add_argument("images", nargs="+", help="图片路径（可多张）")
    ap.add_argument("-p", "--prompt", default="请详细描述这张图片的内容。",
                    help="对图片的提问（默认: 详细描述内容）")
    ap.add_argument("-m", "--model", default=None, help="视觉模型 ID（默认随端点自动选择）")
    ap.add_argument("--base-url", default=DEFAULT_URL,
                    help=f"API 端点（SiliconFlow: {SILICONFLOW_URL}，默认智谱）")
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="最大输出 token（智谱 glm-4v-flash 上限 1024）")
    ap.add_argument("--api-key", default=None,
                    help="API Key（默认读 SILICONFLOW_API_KEY / ZHIPU_API_KEY 环境变量）")
    args = ap.parse_args()

    # key 优先级：显式 --api-key > 按端点自动选（智谱用 ZHIPU，SiliconFlow 用 SILICONFLOW）
    # 避免 SILICONFLOW_API_KEY 失效时连累默认的智谱端点
    if args.api_key:
        key = args.api_key
    elif "siliconflow" in args.base_url:
        key = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get("ZHIPU_API_KEY")
    else:
        key = os.environ.get("ZHIPU_API_KEY") or os.environ.get("SILICONFLOW_API_KEY")
    if not key:
        print("缺少 API Key：请设置 SILICONFLOW_API_KEY / ZHIPU_API_KEY 或用 --api-key",
              file=sys.stderr)
        sys.exit(1)

    # 模型没指定时按 base-url 自动选
    if args.model is None:
        args.model = SILICONFLOW_MODEL if "siliconflow" in args.base_url else DEFAULT_MODEL

    imgs = [Path(p) for p in args.images]
    for im in imgs:
        if not im.exists():
            print(f"图片不存在: {im}", file=sys.stderr)
            sys.exit(1)

    content = [{"type": "text", "text": args.prompt}]
    for im in imgs:
        content.append({"type": "image_url",
                        "image_url": {"url": image_to_data_url(im)}})

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": args.max_tokens,
    }
    req = urllib.request.Request(
        args.base_url,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        print(data["choices"][0]["message"]["content"])
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"API 错误 {e.code}: {body[:500]}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
