import os
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
from rag_engine import RAGSystem

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Bengali-English RAG API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None
    use_agent: bool = False  

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    conversation_id: str
    confidence_score: float

class EvaluationRequest(BaseModel):
    query: str
    expected_answer: str

class EvaluationResponse(BaseModel):
    query: str
    expected_answer: str
    generated_answer: str
    groundedness_score: float
    relevance_score: float
    answer_similarity: float
    retrieved_contexts: List[str]

# Initialize RAG system
rag_system = RAGSystem()

# API Endpoints
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Query the RAG system"""
    try:
        if request.use_agent:
            result = rag_system.agent_query(
                query=request.query,
                conversation_id=request.conversation_id
            )
        else:
            result = rag_system.direct_query(
                query=request.query,
                conversation_id=request.conversation_id
            )
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_endpoint(request: EvaluationRequest):
    """Evaluate the RAG system"""
    try:
        result = rag_system.evaluate_response(
            query=request.query,
            expected_answer=request.expected_answer
        )
        return EvaluationResponse(**result)
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index")
async def index_document(pdf_path: str):
    """Index a PDF document"""
    try:
        rag_system.build_index(pdf_path)
        return {"message": "Document indexed successfully"}
    except Exception as e:
        logger.error(f"Indexing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history"""
    if conversation_id not in rag_system.conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    history = rag_system.conversations[conversation_id]["history"]
    return {
        "conversation_id": conversation_id,
        "history": [{"role": "human" if i % 2 == 0 else "assistant", 
                    "content": msg.content} 
                   for i, msg in enumerate(history)]
    }

if __name__ == "__main__":
    # Start API server
    uvicorn.run(app, host="0.0.0.0", port=8000)