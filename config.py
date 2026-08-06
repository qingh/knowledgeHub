import os

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHROMA_DB_PATH = os.environ.get(
    "CHROMA_DB_PATH",
    os.path.join(BASE_DIR, "data", "local-chroma-data"),  # 默认值
)

# EMBED_MODEL_NAME=BAAI/bge-large-zh-v1.5
EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# 写入与检索必须使用同一个 collection_name，否则会检索到空集合
COLLECTION_NAME = "knowledge_hub"
