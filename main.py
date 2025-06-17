from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
from pinecone_utils import query_pinecone, upload_to_pinecone, save_reply_to_pinecone

app = FastAPI()

# CORS 支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境请改为指定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"message": "Pinecone Semantic Search API is running"}

@app.get("/search")
def search_email(query: str = Query(..., description="用户查询的问题"), top_k: int = 5):
    try:
        print(f"🔍 接收到查询: {query}")
        results = query_pinecone(query, top_k=top_k)
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
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

@app.post("/upsert")
def upsert_vectors(vectors: List[Dict[str, Any]] = Body(..., description="向量列表，每个包含 id, values, metadata")):
    try:
        upload_to_pinecone(vectors)
        return {"message": f"成功上传 {len(vectors)} 条向量"}
    except Exception as e:
        print(f"❌ 上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

@app.post("/save-reply")
def save_reply(reply: Dict[str, Any] = Body(...)):
    try:
        save_reply_to_pinecone(reply)
        return {"message": "✅ 成功写入 Pinecone"}
    except Exception as e:
        print(f"❌ 写入失败: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")