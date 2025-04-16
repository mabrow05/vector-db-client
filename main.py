from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import uvicorn

app = FastAPI(title="Semantic Search Service")

# Initialize Qdrant client
qdrant_client = QdrantClient(host="localhost", port=6333)

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create collection if it doesn't exist
try:
    qdrant_client.create_collection(
        collection_name="text_chunks",
        vectors_config=models.VectorParams(
            size=384,  # Vector size for all-MiniLM-L6-v2
            distance=models.Distance.COSINE
        )
    )
except Exception as e:
    print(f"Collection might already exist: {e}")

class TextChunk(BaseModel):
    text: str
    metadata: Optional[dict] = None

class SearchQuery(BaseModel):
    query: str
    limit: Optional[int] = 5

@app.post("/add_chunk")
async def add_chunk(chunk: TextChunk):
    try:
        # Generate embedding for the text
        vector = model.encode(chunk.text).tolist()
        
        # Add point to Qdrant
        qdrant_client.upsert(
            collection_name="text_chunks",
            points=[
                models.PointStruct(
                    id=hash(chunk.text),  # Simple hash as ID
                    vector=vector,
                    payload={"text": chunk.text, **(chunk.metadata or {})}
                )
            ]
        )
        return {"status": "success", "message": "Chunk added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search(query: SearchQuery):
    try:
        # Generate embedding for the search query
        query_vector = model.encode(query.query).tolist()
        
        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name="text_chunks",
            query_vector=query_vector,
            limit=query.limit
        )
        
        # Format results
        results = [
            {
                "text": hit.payload["text"],
                "score": hit.score,
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"}
            }
            for hit in search_results
        ]
        
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
