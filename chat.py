import os

from config import CHROMA_DB_PATH, COLLECTION_NAME, EMBED_MODEL_NAME

# 国内网络拉 HuggingFace 模型慢，走镜像。
# 注意：必须在 import huggingface_hub / langchain_huggingface 之前设置，
# 否则 huggingface_hub 在 import 时就已把 HF_ENDPOINT 读进常量缓存，改了也不生效。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 模型已下载到本地缓存后，开启离线模式：不再联网发 HEAD 检查更新，
# 避免网络差时卡在 SSL 握手超时。若模型还没下载，请先注释掉本行下载模型。
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import chainlit as cl
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings

from base import chat

# ---- 全局初始化（只需一次，多个用户会话共用）----
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    encode_kwargs={"normalize_embeddings": True},  # bge 系列官方建议做 L2 normalize
)
vectorstore = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 20},
)

# retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = chat()

prompt = ChatPromptTemplate.from_template("""
你是公司内部的智能问答助手，请仅根据以下检索到的内部文档，用简洁的语言回答用户问题。
如果文档中没有相关信息，请明确告知用户，不要编造答案。

【检索到的文档】
{context}

【用户问题】
{question}
""")


def format_docs(docs):
    return "\n\n".join(
        f"[来源: {doc.metadata.get('source', '未知')}]\n{doc.page_content}"
        for doc in docs
    )


@cl.on_chat_start
async def on_chat_start():
    # 每个用户会话独立保存 chain（如果之后要加会话历史，可以在这里按 session 定制）
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    cl.user_session.set("rag_chain", rag_chain)

    await cl.Message(
        content="你好，我是公司内部知识问答助手，有什么可以帮你查的？"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    rag_chain = cl.user_session.get("rag_chain")
    assert rag_chain is not None, "rag_chain not initialized in user_session"

    # 先单独检索一次文档，用来展示来源（chain 内部也会检索，这里额外取一次仅用于展示）
    docs = retriever.invoke(message.content)

    msg = cl.Message(content="")

    async for chunk in rag_chain.astream(
        message.content,
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    # 把引用来源作为 Element 附加在消息末尾，方便溯源
    if docs:
        from collections import Counter

        source_counts = Counter(d.metadata.get("source", "未知来源") for d in docs)
        sources_text = "\n".join(
            f"- {s}（命中 {c} 处相关内容）" for s, c in source_counts.items()
        )
        msg.content += f"\n\n---\n**参考来源：**\n{sources_text}"

    await msg.send()
