# 企业内部知识库 + AI 问答助手

## Requirements
- Python 3.12.0
- 依赖见 `requirements.txt`

一个基于 **RAG（检索增强生成）** 的企业内部知识库问答系统。把公司内部的 PDF 文档（如员工手册、规章制度）灌入本地向量库，再通过带来源引用的对话界面回答员工提问，答案严格基于内部文档，不编造内容。

## 功能特性

- **本地文档入库**：批量加载 `doc/` 目录下的 PDF，自动切分并向量化存入 Chroma
- **中文语义检索**：使用 `BAAI/bge-small-zh-v1.5` 中文 Embedding 模型，针对中文场景优化
- **MMR 检索策略**：从 20 个候选中选出 6 个兼顾相关性与多样性的片段，减少重复内容占用上下文
- **流式对话界面**：基于 Chainlit 的 Web 聊天 UI，回答逐字输出
- **来源可溯**：每次回答末尾附带引用的文档来源及命中次数
- **拒绝幻觉**：Prompt 约束模型仅依据检索结果作答，文档中没有的内容会明确告知用户
- **模型可替换**：LLM 通过 OpenAI 兼容接口配置，可接入 OpenAI / DeepSeek / 通义千问 / 本地 Ollama 等
- **国内网络友好**：默认走 HuggingFace 镜像站，模型下载后可开启离线模式避免网络卡顿

## 技术栈

| 分层 | 选型 |
| --- | --- |
| 应用框架 | Chainlit 2.x |
| 编排框架 | LangChain 1.x（LCEL） |
| 大语言模型 | 任意 OpenAI 兼容接口（`langchain-openai`） |
| Embedding | BAAI/bge-small-zh-v1.5（`langchain-huggingface` + sentence-transformers） |
| 向量数据库 | Chroma（本地持久化） |
| 文档解析 | PyMuPDF |
| 运行环境 | Python 3.12.0 |

## 目录结构

```
knowledge-hub/
├── doc/                    # 待入库的 PDF 源文档（已 gitignore）
├── src/
│   ├── __init__.py
│   ├── base.py             # LLM 客户端构建，读取 .env 配置并缓存实例
│   ├── ingest.py           # 离线入库脚本：加载 → 切分 → 向量化 → 写入 Chroma
│   └── chat.py             # Chainlit 应用：RAG 检索问答 + 流式输出 + 来源展示
├── config.py               # 公共配置：CHROMA_DB_PATH、EMBED_MODEL_NAME 等
├── data/local-chroma-data  # Chroma 持久化默认目录（自动生成，已 gitignore）
├── .env                    # 环境变量（已 gitignore）
└── readme.md
```

## 工作流程

```
PDF 文档 ──► DirectoryLoader ──► RecursiveCharacterTextSplitter ──► bge Embedding ──► Chroma
                                                                                        │
用户提问 ─────────────────────────────► MMR 检索 (k=6, fetch_k=20) ◄────────────────────┘
                                              │
                                              ▼
                                    Prompt 拼装 ──► LLM ──► 流式回答 + 来源引用
```

## 快速开始

### 1. 环境准备

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install chainlit langchain langchain-community langchain-openai \
            langchain-huggingface langchain-chroma langchain-text-splitters \
            chromadb sentence-transformers pymupdf python-dotenv
```

### 2. 配置环境变量

在项目根目录创建 `.env`：

```ini
MODEL=deepseek-chat                        # 模型名称
BASE_URL=https://api.deepseek.com/v1       # OpenAI 兼容接口地址
API_KEY=sk-xxxxxxxxxxxxxxxx                # 你的 API Key
CHROMA_DB_PATH=/home/my-chroma-data        # chromadb data
```

三个变量缺一不可，缺失时启动会直接报错提示。

### 3. 放入文档并建库

把 PDF 放进 `doc/` 目录，然后在项目根目录执行：

```bash
python -m src.ingest
```

> 注意：脚本使用相对路径 `./doc` 读取文档，必须在**项目根目录**下运行。
> 首次运行会从 HuggingFace 镜像下载 Embedding 模型（约 100MB）。
> 若下载失败，请先注释掉 `ingest.py` 中的 `HF_HUB_OFFLINE` 那一行再试。

输出示例：

```
已加载 12 个文档片段
已切分为 148 个 chunk
入库完成，共 148 个 chunk
```

### 4. 启动问答服务

```bash
chainlit run src/chat.py -w
```

浏览器打开 http://localhost:8000 即可开始提问。`-w` 为热重载，开发时使用。

## 关键参数说明

| 参数 | 位置 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `chunk_size` | `ingest.py` | 500 | 单个 chunk 字符数，适配 bge 的 512 token 上限 |
| `chunk_overlap` | `ingest.py` | 80 | 相邻 chunk 重叠字符数，避免句意被切断 |
| `separators` | `ingest.py` | 中文标点优先 | 按段落 → 换行 → 句号 → 逗号逐级切分 |
| `search_type` | `chat.py` | `mmr` | 最大边际相关性，兼顾相关度与多样性 |
| `k` / `fetch_k` | `chat.py` | 6 / 20 | 召回 20 条候选，最终取 6 条送入 Prompt |
| `DEFAULT_TEMPERATURE` | `base.py` | 0 | 温度为 0，保证回答稳定、减少发挥 |
| `DEFAULT_MAX_TOKENS` | `base.py` | 1024 | 单次回答最大输出长度 |

## 常见问题

**Q：文档更新后需要做什么？**
重新运行 `python -m src.ingest`。当前实现为追加写入，若需完全重建，请先删除 `data/local-chroma-data/` 目录。实际路径以 `.env` 中 `CHROMA_DB_PATH` 为准。

**Q：想换更强的 Embedding 模型？**
打开 `config.py`，把 `EMBED_MODEL_NAME` 改为 `BAAI/bge-large-zh-v1.5`（取消上方注释即可，`ingest.py` 和 `chat.py` 会自动读取）。切换后必须重建向量库，因为模型维度变了。

**Q：模型下载卡住 / SSL 超时？**
代码默认设置 `HF_ENDPOINT=https://hf-mirror.com` 走国内镜像，并开启 `HF_HUB_OFFLINE=1` 跳过联网检查。首次下载模型时需临时注释掉 `HF_HUB_OFFLINE` 那一行。

**Q：回答说「文档中没有相关信息」？**
说明检索没有命中相关片段。可以尝试调大 `k`、换用 `bge-large-zh-v1.5`，或检查 PDF 是否为扫描件（图片型 PDF 无法直接提取文字，需要先做 OCR）。

**Q：Windows 终端中文乱码？**
`ingest.py` 已自动将 stdout 重配置为 UTF-8，无需额外处理。

## 后续规划

- [ ] 多轮对话记忆（当前每次提问相互独立）
- [ ] 支持 Word / Markdown / TXT 等更多文档格式
- [ ] 增量入库与去重，避免重复运行产生冗余向量
- [ ] 引用来源精确到页码并支持原文跳转
- [ ] 用户登录与文档权限隔离
