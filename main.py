from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pinecone_utils import query_pinecone

app = FastAPI()

# CORS 中间件支持（如果用 n8n 或 Web 前端访问 API）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 可以设置为你的前端地址
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
                    "email_id": r.metadata.get("email_id", ""),
                    "subject": r.metadata.get("subject", ""),
                    "summary": r.metadata.get("email_summary", ""),
                    "issue_type": r.metadata.get("issue_type", ""),
                    "ideal_reply": r.metadata.get("ideal_reply", "")
                }
                for r in results
            ]
        }
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
