"""
FastAPI application for RAG backend with ChromaDB.
"""
from fastapi import FastAPI, HTTPException

from .models import CodeIngestRequest, QueryRequest
from .database import (
    get_collection,
    get_client,
    get_collection_name,
    reset_collection
)
from .chunker import chunk_python_code
from .search import perform_vector_search, perform_hybrid_search


app = FastAPI(title="RAG Backend with ChromaDB")


@app.post("/ingest")
async def ingest_code(request: CodeIngestRequest):
    """ingest python code, chunk it via ast, and store in chromadb"""
    try:
        chunks = chunk_python_code(request.code, request.filepath, request.repo_name)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="no valid code chunks found")
        
        # prepare batch insertion
        ids = [chunk["id"] for chunk in chunks]
        documents = [chunk["content"] for chunk in chunks]
        
        # filter out None values from metadata - chromadb cannot handle None in metadata fields
        metadatas = []
        for chunk in chunks:
            filtered_metadata = {}
            for key, value in chunk["metadata"].items():
                if value is not None:  # only include non-None values
                    filtered_metadata[key] = value
            metadatas.append(filtered_metadata)
        
        # chromadb will automatically generate embeddings using the integrated function
        collection = get_collection()
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        return {
            "status": "success",
            "chunks_ingested": len(chunks),
            "collection_name": get_collection_name()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ingestion failed: {str(e)}")


@app.post("/query")
async def query_code(request: QueryRequest):
    """
    query the vector db using vector similarity search.
    for hybrid search (vector + bm25), use the /query-hybrid endpoint.
    """
    try:
        # perform vector similarity search
        vector_results = perform_vector_search(request.query, request.n_results)
        
        # format results
        formatted_results = []
        if vector_results.get("ids") and vector_results["ids"] and vector_results["ids"][0]:
            for i, doc_id in enumerate(vector_results["ids"][0]):
                formatted_results.append({
                    "id": doc_id,
                    "content": vector_results["documents"][0][i],
                    "metadata": vector_results["metadatas"][0][i],
                    "distance": vector_results["distances"][0][i]
                })
        
        return {
            "status": "success",
            "query": request.query,
            "results": formatted_results,
            "count": len(formatted_results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"query failed: {str(e)}")


@app.post("/query-hybrid")
async def query_hybrid(request: QueryRequest):
    """
    hybrid search combining vector similarity and bm25 keyword search.
    this provides better results by leveraging both semantic and lexical matching.
    """
    try:
        sorted_results = perform_hybrid_search(request.query, request.n_results)
        
        return {
            "status": "success",
            "query": request.query,
            "results": sorted_results,
            "count": len(sorted_results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"hybrid search failed: {str(e)}")


@app.get("/health")
async def health_check():
    """check chromadb connection health"""
    try:
        # test chromadb connection
        client = get_client()
        heartbeat = client.heartbeat()
        
        return {
            "status": "healthy",
            "chroma_heartbeat": heartbeat,
            "collection_name": get_collection_name()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"chroma db unavailable: {str(e)}")


@app.delete("/collection")
async def delete_collection():
    """delete the entire collection (use with caution)"""
    try:
        reset_collection()
        return {
            "status": "success",
            "message": f"collection '{get_collection_name()}' reset"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"collection reset failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        timeout_keep_alive=75,
        timeout_graceful_shutdown=10
    )
