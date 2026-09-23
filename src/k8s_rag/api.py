"""
FastAPI endpoints for K8s RAG.
"""

from typing import Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str
    sources: list[str]
    confidence: float


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    rag_ready: bool
    vector_store_size: int
    llm_loaded: bool


class K8sRAGAPI:
    """
    FastAPI application for K8s RAG.
    
    Provides:
    - /health - Health check endpoint
    - /query - Query the RAG system
    - /metrics - Get evaluation metrics
    """
    
    def __init__(
        self,
        rag_chain,
        vectorstore,
        host: str = "0.0.0.0",
        port: int = 8000,
        enable_cors: bool = True
    ):
        """
        Initialize the API.
        
        Args:
            rag_chain: RAGChain instance
            vectorstore: VectorStoreManager instance
            host: Host to bind the API
            port: Port to bind the API
            enable_cors: Whether to enable CORS
        """
        self.rag_chain = rag_chain
        self.vectorstore = vectorstore
        
        # Create FastAPI app
        self.app = FastAPI(
            title="K8s RAG API",
            description="A production-ready RAG system for Kubernetes documentation",
            version="0.2.0"
        )
        
        # Add CORS middleware if enabled
        if enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Add endpoints
        self._add_endpoints()
        
        # Store for background tracking
        self._health_data = {
            "status": "healthy",
            "rag_ready": rag_chain is not None,
            "vector_store_size": vectorstore.document_count if vectorstore else 0,
            "llm_loaded": hasattr(rag_chain, 'llm') and rag_chain.llm is not None,
        }
    
    def _add_endpoints(self):
        """Add API endpoints."""
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy",
                rag_ready=self._health_data["rag_ready"],
                vector_store_size=self._health_data["vector_store_size"],
                llm_loaded=self._health_data["llm_loaded"],
            )
        
        @self.app.post("/query", response_model=QueryResponse)
        async def query(request: QueryRequest):
            """Query the RAG system."""
            if not self._health_data["rag_ready"]:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="RAG system not ready. Please initialize the system first."
                )
            
            # Get answer with sources
            result = self.rag_chain.with_sources(request.question)
            
            return QueryResponse(
                answer=result["answer"],
                sources=result["sources"],
                confidence=self._calculate_confidence(result["answer"], result["sources"]),
            )
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get evaluation metrics."""
            return {
                "factual_accuracy": self._health_data.get("factual_accuracy", 0),
                "hallucination_rate": self._health_data.get("hallucination_rate", 0),
                "total_questions_tested": self._health_data.get("total_questions", 0),
            }
        
        @self.app.get("/docs")
        async def api_docs():
            """Serve interactive API documentation (Swagger UI)."""
            return self.app
    
    def _calculate_confidence(self, answer: str, sources: list[str]) -> float:
        """
        Calculate confidence score based on answer quality.
        
        Args:
            answer: Generated answer
            sources: List of source URLs
        
        Returns:
            Confidence score (0-1)
        """
        if not sources:
            return 0.5
        
        # Check if answer has substantial content
        if len(answer.strip()) < 20:
            return 0.3
        
        # Check if sources are from official K8s docs
        from urllib.parse import urlparse
        official_sources = 0
        for source in sources:
            if "kubernetes.io" in source or "k8s_rag_data" in source:
                official_sources += 1
        
        # Calculate confidence based on number of official sources
        confidence = official_sources / len(sources)
        
        # Boost confidence if answer is detailed
        word_count = len(answer.split())
        if word_count > 50:
            confidence = min(1.0, confidence + 0.2)
        
        return min(1.0, confidence)
    
    def run(self):
        """Run the API server."""
        uvicorn.run(
            self.app,
            host=self.app.config.get("host", "0.0.0.0"),
            port=self.app.config.get("port", 8000),
            reload=False
        )
    
    async def run_async(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the API server asynchronously."""
        import asyncio
        
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            reload=False
        )
        server = uvicorn.Server(config)
        
        async with server:
            await server.serve()


def create_api(
    rag_chain,
    vectorstore,
    host: str = "0.0.0.0",
    port: int = 8000,
    enable_cors: bool = True
) -> K8sRAGAPI:
    """
    Create a K8s RAG API instance.
    
    Args:
        rag_chain: RAGChain instance
        vectorstore: VectorStoreManager instance
        host: Host to bind the API
        port: Port to bind the API
        enable_cors: Whether to enable CORS
    
    Returns:
        K8sRAGAPI instance
    """
    return K8sRAGAPI(
        rag_chain=rag_chain,
        vectorstore=vectorstore,
        host=host,
        port=port,
        enable_cors=enable_cors
    )
