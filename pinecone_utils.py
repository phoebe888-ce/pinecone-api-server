import os
import httpx
from typing import List, Dict
from uuid import uuid4
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
# 加载环境变量
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")  # 默认值为 us-east-1, "us-east-1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")





# 初始化 Pinecone 客户端
pc = Pinecone(api_key=PINECONE_API_KEY)

# 自动创建索引（如不存在）
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    print(f"📌 正在创建索引: {PINECONE_INDEX_NAME}")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
    )

# 获取索引
index = pc.Index(PINECONE_INDEX_NAME)

# 初始化 OpenAI 客户端，添加超时控制和重试机制
http_client = httpx.Client(timeout=10.0)
openai_client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)


def get_embedding(text: str) -> List[float]:
    """调用 OpenAI Embedding API 获取文本向量"""
    try:
        response = openai_client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        if len(embedding) != 1536:
            raise ValueError("返回嵌入维度异常")
        return embedding
    except Exception as e:
        print(f"❌ 获取嵌入失败: {e}")
        return []  # 返回空表示失败


def upload_to_pinecone(data: List[Dict]):
    """
    向 Pinecone 上传向量数据
    :param data: List of dicts, each must contain `id`, `text`, and optional `metadata`
    """
    vectors = []
    for item in data:
        embedding = get_embedding(item["text"])
        if not embedding:
            print(f"⚠️ 跳过嵌入失败项: {item.get('id', '[无ID]')}")
            continue

        vector_id = item.get("id", str(uuid4()))
        metadata = item.get("metadata", {})

        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": metadata
        })

    if vectors:
        try:
            index.upsert(vectors=vectors)
            print(f"✅ 已上传 {len(vectors)} 条向量到 Pinecone")
        except Exception as e:
            print(f"❌ 向 Pinecone 上传失败: {e}")
    else:
        print("🚫 无有效向量可上传")


def query_pinecone(query_text: str, top_k: int = 5) -> List[Dict]:
    """查询 Pinecone 中最相似向量"""
    embedding = get_embedding(query_text)
    if not embedding:
        return []  # 嵌入失败
    results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    return results.matches




