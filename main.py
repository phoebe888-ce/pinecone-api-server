from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

from pinecone_utils import query_pinecone, upload_to_pinecone, save_reply_to_pinecone, index

app = FastAPI()

# 🚨 生产环境请限制 allow_origins 范围
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ✅ 请求体模型定义
class UpsertVector(BaseModel):
    id: str
    values: List[float]
    metadata: Dict[str, Any]

class SaveReply(BaseModel):
    threadId: str
    customerMsg: str
    aiReply: str
    timestamp: str

class UpdateReplyRequest(BaseModel):
    threadId: str
    aiReply: str

@app.get("/")
def health_check():
    return {"message": "✅ Pinecone Semantic Search API is running"}

@app.get("/search")
def search_email(query: str = Query(..., description="用户查询的问题"), top_k: int = 5):
    try:
        clean_query = query.strip().replace("\n", " ")
        if not clean_query:
            raise HTTPException(status_code=400, detail="Query must not be empty.")
        print(f"🔍 接收到查询: {clean_query}")

        results = query_pinecone(clean_query, top_k=top_k)
        print(f"✅ 查询成功，返回 {len(results)} 条结果")

        return {
            "matches": [
                {
                    "id": r.id,
                    "score": r.score,
                    "threadId": r.metadata.get("threadId", ""),
                    "customerMsg": r.metadata.get("customerMsg", ""),
                    "aiReply": r.metadata.get("aiReply", ""),
                    "timestamp": r.metadata.get("timestamp", "")
                }
                for r in results
            ]
        }
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        raise HTTPException(status_code=500, detail="A server error has occurred")

@app.post("/upsert")
def upsert_vectors(vectors: List[UpsertVector]):
    try:
        upload_to_pinecone([v.dict() for v in vectors])
        return {"message": f"✅ 成功上传 {len(vectors)} 条向量"}
    except Exception as e:
        print(f"❌ 上传失败: {e}")
        raise HTTPException(status_code=500, detail="上传失败")

@app.post("/save-reply")
def save_reply(reply: SaveReply):
    try:
        save_reply_to_pinecone(reply.dict())
        return {"message": "✅ 成功写入 Pinecone"}
    except Exception as e:
        print(f"❌ 写入失败: {e}")
        raise HTTPException(status_code=500, detail="写入失败")

@app.patch("/update-reply")
def update_reply(data: UpdateReplyRequest):
    try:
        existing = index.fetch(ids=[data.threadId])
        if data.threadId in existing.vectors:
            old_vector = existing.vectors[data.threadId]
            updated_metadata = old_vector.metadata
            updated_metadata["aiReply"] = data.aiReply

            index.upsert([
                (data.threadId, old_vector.values, updated_metadata)
            ])
            return {"message": "✅ 成功更新回复"}
        else:
            return {"message": "⚠️ 未找到指定 threadId"}
    except Exception as e:
        print(f"❌ 更新失败: {e}")
        raise HTTPException(status_code=500, detail="更新失败")