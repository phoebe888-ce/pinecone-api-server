import os
import httpx
from typing import List, Dict
from uuid import uuid4
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    print(f"📌 正在创建索引: {PINECONE_INDEX_NAME}")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
    )

index = pc.Index(PINECONE_INDEX_NAME)
http_client = httpx.Client(timeout=10.0)
openai_client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)

# def get_embedding(text: str) -> List[float]:
#     try:
#         response = openai_client.embeddings.create(
#             input=[text],
#             model=""
#                   ""
#         )
#         embedding = response.data[0].embedding
#         if len(embedding) != 1536:
#             raise ValueError("返回嵌入维度异常")
#         return embedding
#     except Exception as e:
#         print(f"❌ 获取嵌入失败: {e}")
#         return []

# def upload_to_pinecone(data: List[Dict]):
#     vectors = []
#     for item in data:
#         embedding = get_embedding(item["text"])
#         if not embedding:
#             print(f"⚠️ 跳过嵌入失败项: {item.get('id', '[无ID]')}")
#             continue
#
#         vector_id = item.get("id", str(uuid4()))
#         metadata = item.get("metadata", {})
#
#         vectors.append({
#             "id": vector_id,
#             "values": embedding,
#             "metadata": metadata
#         })
#
#     if vectors:
#         try:
#             index.upsert(vectors=vectors)
#             print(f"✅ 已上传 {len(vectors)} 条向量到 Pinecone")
#         except Exception as e:
#             print(f"❌ 向 Pinecone 上传失败: {e}")
#     else:
#         print("🚫 无有效向量可上传")

def upload_to_pinecone(data: List[Dict]):
    vectors = []
    for item in data:
        embedding = item.get("embedding")
        if not embedding or not isinstance(embedding, list):
            print(f"⚠️ 跳过无效或缺失 embedding 的项: {item.get('id', '[无ID]')}")
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


def query_pinecone(query_text: str, top_k: int = 5):
    embedding = get_embedding(query_text)
    if not embedding:
        return []
    results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
    return results.matches

def save_reply_to_pinecone(data: Dict[str, str]):
    thread_id = data.get("threadId", str(uuid4()))
    customer_msg = data.get("customerMsg", "")
    ai_reply = data.get("aiReply", "")
    timestamp = data.get("timestamp") or datetime.utcnow().isoformat()
    embedding = data.get("embedding")

    if not embedding:
        raise ValueError("必须提供 embedding 字段")

    # 确认 embedding 是列表，且长度合理
    if not isinstance(embedding, list) or len(embedding) != 1536:
        raise ValueError("embedding 格式不正确或维度异常")

    index.upsert(vectors=[{
        "id": thread_id,
        "values": embedding,
        "metadata": {
            "threadId": thread_id,
            "customerMsg": customer_msg,
            "aiReply": ai_reply,
            "timestamp": timestamp
        }
    }])
    print(f"✅ 已保存语义回复向量 threadId={thread_id}")
