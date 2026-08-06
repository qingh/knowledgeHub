import os

# 国内网络拉 HuggingFace 模型慢，走镜像。
# 注意：必须在 import huggingface_hub / langchain_huggingface 之前设置，
# 否则 huggingface_hub 在 import 时就已把 HF_ENDPOINT 读进常量缓存，改了也不生效。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 模型已下载到本地缓存后，开启离线模式：不再联网发 HEAD 检查更新，
# 避免网络差时卡在 SSL 握手超时。若模型还没下载，请先注释掉本行下载模型。
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import io
import sys

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHROMA_DB_PATH, COLLECTION_NAME, EMBED_MODEL_NAME

# Windows 终端默认 GBK，打印中文会乱码。
# sys.stdout 的静态类型是 TextIO（没有 reconfigure），只有具体实现 TextIOWrapper 才有；
# 被重定向到管道/StringIO 时也可能不是 TextIOWrapper，所以先做 isinstance 判断。
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8")


def build_vectorstore():
    # ========== 1. 加载 PDF ==========
    loader = DirectoryLoader(
        "./doc",
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,  # type: ignore[arg-type]
    )

    try:
        documents = loader.load()
    except (FileNotFoundError, RuntimeError) as e:
        print(f"加载文档失败: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"已加载 {len(documents)} 个文档片段")

    # ========== 2. 切分 ==========
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 配合 bge-large-zh 的 512 token 上限比较合适
        chunk_overlap=80,  # 重叠多一些，避免句意被切碎
        separators=["\n\n", "\n", "。", "！", "？", "，", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"已切分为 {len(chunks)} 个 chunk")

    # ========== 3. Embedding ==========
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},  # bge 系列官方建议做 L2 normalize
    )

    # ========== 4. 存入 Chroma（避免重复写入） ==========
    Chroma.from_documents(
        database="my_hub",
        collection_name=COLLECTION_NAME,
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
    )

    print(f"入库完成，共 {len(chunks)} 个 chunk")


if __name__ == "__main__":
    build_vectorstore()
