# content-distillery

**把「素材」变成「文本」的 AI 工具集** — 识图 / 扫描书 OCR / 视频转写。

Turn any material into text — images, scanned books and videos, ready for downstream LLM / RAG pipelines.

---

一套从个人知识管道里长出来的小工具，每个都解决一个具体问题：

| 工具 | 干什么 | 输入 → 输出 | 依赖 |
|---|---|---|---|
| [`vision.py`](vision.py) | 让纯文本模型"看图" | 图片 → 文字描述 | **零依赖**（纯标准库） |
| [`ocr_pdf.py`](ocr_pdf.py) | 扫描版 PDF 转文本 | PDF → 分页 txt | PyMuPDF + RapidOCR |
| [`skills/video-downloader`](skills/video-downloader/) | 视频归档 + 转写 | 视频链接 → 视频 + 原文案 + 音频 + 转写稿 | yt-dlp + ffmpeg |

> 设计目标：把图片、书、视频这些"非文本素材"统一转成文本，喂给 LLM、知识库或 RAG 检索——这是构建个人 AI 知识管道的第一步。

## 快速开始

```bash
git clone https://github.com/18027972083/content-distillery.git
cd content-distillery
pip install -r requirements.txt        # 仅 ocr_pdf 需要；vision.py 零依赖
```

---

## vision.py — 识图（零依赖）

把图片交给任意 OpenAI 兼容的视觉模型，拿回文字描述。适合给"看不见图"的纯文本模型 / Agent 补充视觉输入。

**默认走智谱 `glm-4v-flash`（有免费额度），也可切到 SiliconFlow 或任何兼容端点。**

```bash
# 默认：智谱
export ZHIPU_API_KEY="your-key"
python vision.py photo.jpg -p "描述这张图的内容"

# 多张图 + 自定义问题
python vision.py a.jpg b.png -p "这两张图的 UI 布局差异是什么？"

# 换端点（SiliconFlow）
python vision.py photo.jpg \
    --base-url https://api.siliconflow.cn/v1/chat/completions \
    --max-tokens 1024
```

| 参数 | 说明 |
|---|---|
| `-p/--prompt` | 对图片的提问（默认：详细描述内容） |
| `--base-url` | API 端点（默认智谱；SiliconFlow 见示例） |
| `-m/--model` | 视觉模型 ID（默认随端点自动选择） |
| `--api-key` | 显式传 key（默认读环境变量 `ZHIPU_API_KEY` / `SILICONFLOW_API_KEY`） |
| `--max-tokens` | 最大输出 token（智谱 glm-4v-flash 上限 1024） |

**为什么值得一看**：整个脚本只用 Python 标准库（`urllib` + `base64`），没有任何第三方依赖——拷到任何有 Python 的机器上就能跑。

---

## ocr_pdf.py — 扫描版 PDF 转文本

PyMuPDF 渲染页面 → RapidOCR（ONNX 本地推理）识别 → 输出分页文本。**全程本地，不联网**。

```bash
pip install -r requirements.txt

python ocr_pdf.py scanned.pdf                    # 全本，默认 300 DPI
python ocr_pdf.py scanned.pdf --pages 1-20       # 只处理前 20 页
python ocr_pdf.py scanned.pdf -o out.txt --dpi 240 --pages 5,10-12
```

| 参数 | 说明 |
|---|---|
| `-o/--output` | 输出路径（默认：同目录同名 `.ocr.txt`） |
| `--dpi` | 渲染分辨率（默认 300，扫描质量差可提高） |
| `--pages` | 页码范围：`1-20`、`3`、`5,10-12` |

输出为分页文本（`===== 第 N 页 =====` 分隔），可直接接下游清洗或入库。

---

## skills/video-downloader — 视频归档 + 转写

一个 **AI agent skill 格式**的工具（`SKILL.md` + 可独立运行的 CLI）：给一个视频链接，拿回一个素材文件夹。

**支持平台**：抖音（H5 主路由 + yt-dlp 回退）、B 站、YouTube、小红书。视频号暂未实现（见 SKILL.md 说明）。

```bash
cd skills/video-downloader
python3 scripts/download_video.py "https://v.douyin.com/xxxx" --output-dir ./downloads
```

产出结构：

```
downloads/<platform>_<id>/
├── video.mp4            # 视频文件
├── post_caption.txt     # 平台原文案（标题 / 正文 / 标签）
├── audio.m4a|mp3        # 抽取的音频
├── transcript.txt       # 语音转写稿（ASR）
└── metadata.json        # 归一化元数据 + 下载/转写状态
```

转写（ASR）两种后端：

```bash
# 中文语音推荐：SiliconFlow SenseVoiceSmall
SILICONFLOW_API_KEY="..." python3 scripts/download_video.py "<url>" \
    --output-dir ./downloads --asr siliconflow --asr-language Chinese

# 本地 Whisper
python3 scripts/download_video.py "<url>" --output-dir ./downloads --asr whisper --asr-model small
```

- `--asr auto`（默认）：有 `SILICONFLOW_API_KEY` 走云端，否则用本地 `whisper`
- `--asr none`：只下载不转写
- `--metadata-only`：只抓文案和元数据（秒回）
- 需要 `ffmpeg`（抽音频）、`yt-dlp`（部分平台）、可选本地 `whisper`

新增平台：在 `scripts/providers/` 加一个模块（实现 `PLATFORM` / `supports()` / `fetch()`）并在 `__init__.py` 注册即可，产出物契约保持不变。

---

## 目录结构

```
content-distillery/
├── vision.py                  # 识图（零依赖）
├── ocr_pdf.py                 # 扫描 PDF OCR
├── requirements.txt
└── skills/
    └── video-downloader/      # agent skill：视频下载 + 转写
        ├── SKILL.md
        └── scripts/
            ├── download_video.py
            ├── asr.py
            └── providers/     # douyin / bilibili / youtube / xiaohongshu
```

## Roadmap

- **v0.1**（当前）— 三件套：识图 / OCR / 视频转写
- **v0.2** — RAG 知识库：把转出的文本入库、切分、检索、问答（本仓库的下一站）
- **v0.3** — MCP Server 封装：让 AI Agent 直接调用这些能力

## 使用边界

- `video-downloader` 仅用于下载你有权保存、或在授权范围内可归档的内容；不用于绕过 DRM、付费墙或平台权限做未授权分发。
- 转写/识别的结果质量取决于模型与音视频质量，重要用途请人工校对。

## License

[MIT](LICENSE)
