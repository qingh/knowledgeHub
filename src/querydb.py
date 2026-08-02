import os

import chromadb

from config import BASE_DIR

path = os.path.join(BASE_DIR, "chroma_db")

client = chromadb.PersistentClient(path=path)

print(path, client.list_collections())

collection = client.get_collection(name="knowledge_hub")

print("文档总数：", collection.count())
print("文档总数：", collection.peek())

# result = collection.get(include=["documents", "metadatas"])

# print(result)
