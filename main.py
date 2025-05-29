from fastapi import FastAPI, Query
from pinecone_utils import query_pinecone

app = FastAPI()


@app.get("/")
def health_check():
    return {"message": "Pinecone Semantic Search API is running"}


# @app.get("/search")
# def search_email(query: str = Query(..., description="用户查询的问题"), top_k: int = 5):
#     results = query_pinecone(query, top_k=top_k)
#     return {
#         "matches": [
#             {
#                 "id": r.id,
#                 "score": r.score,
#                 "subject": r.metadata.get("subject", ""),
#                 "summary": r.metadata.get("summary", ""),
#                 "ideal_reply": r.metadata.get("ideal_reply", "")
#             }
#             for r in results
#         ]
#     }
@app.get("/search")
def search_email(query: str = Query(..., description="用户查询的问题"), top_k: int = 5):
    try:
        print(f"🔍 接收到查询: {query}")
        results = query_pinecone(query, top_k=top_k)
        print(f"✅ 查询成功，返回 {len(results)} 条结果")
        # return {"matches": [r.metadata for r in results]}
        return {
                "matches": [
                    {
                        "id": r.id,
                        "score": r.score,
                        "subject": r.metadata.get("subject", ""),
                        "email_id":r.metadata["email_id"],  
                        "summary": r.metadata.get("email_summary", ""),
                        "issue_type":r.metadata.get("issue_type",""),
                        "ideal_reply": r.metadata.get("ideal_reply", "")
                    }
                    for r in results
                ]
            }
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
