"""
Enhanced retriever for K8s RAG with better formatting.
"""

from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.retrievers import Retriever
from langchain_core.runnables import Runnable


class EnhancedRetriever:
    """
    Enhanced retriever that returns both documents and their metadata.
    
    Provides:
    - Similarity search with score threshold filtering
    - Enhanced context formatting with source attribution
    - Metadata preservation
    """
    
    def __init__(
        self,
        vectorstore,
        search_threshold: float = 0.3,
        top_k: int = 3,
        format_context: bool = True
    ):
        """
        Initialize the enhanced retriever.
        
        Args:
            vectorstore: FAISS vector store instance
            search_threshold: Minimum similarity score threshold
            top_k: Maximum number of documents to retrieve
            format_context: Whether to format context with source attribution
        """
        self.vectorstore = vectorstore
        self.search_threshold = search_threshold
        self.top_k = top_k
        self.format_context = format_context
    
    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The search query
        
        Returns:
            List of relevant Document objects
        """
        docs_with_scores = self.vectorstore.similarity_search_with_score(
            query,
            k=self.top_k
        )
        
        # Filter by threshold
        filtered = [
            doc for doc, score in docs_with_scores
            if score >= self.search_threshold
        ][:self.top_k]
        
        return filtered
    
    def retrieve_with_scores(self, query: str) -> List[tuple]:
        """
        Retrieve documents with their similarity scores.
        
        Args:
            query: The search query
        
        Returns:
            List of (Document, score) tuples
        """
        return self.vectorstore.similarity_search_with_score(
            query,
            k=self.top_k
        )
    
    def get_context(self, query: str) -> str:
        """
        Get formatted context from retrieved documents.
        
        Args:
            query: The search query
        
        Returns:
            Formatted context string with source attribution
        """
        docs = self.retrieve(query)
        
        if not docs:
            return ""
        
        context_parts = []
        for doc in docs:
            # Format with source attribution
            source_info = self._format_source(doc)
            content = doc.page_content
            
            if self.format_context:
                context_parts.append(f"{source_info}\n{content}")
            else:
                context_parts.append(content)
        
        return "\n\n".join(context_parts)
    
    def _format_source(self, doc: Document) -> str:
        """
        Format source information from document metadata.
        
        Args:
            doc: Document object
        
        Returns:
            Formatted source string
        """
        source_url = doc.metadata.get("source_url", "")
        source_file = doc.metadata.get("source_file", "")
        title = doc.metadata.get("title", "")
        
        # Prefer URL if available, otherwise use file
        if source_url:
            # Extract just the path from the URL
            from urllib.parse import urlparse
            parsed = urlparse(source_url)
            path = parsed.path
            source = path.rstrip('/').split('/')[-1] if path else "Kubernetes Documentation"
        elif source_file:
            source = source_file
        else:
            source = "Kubernetes Documentation"
        
        return f"[Source: {source}]"
    
    def get_context_dict(self, query: str) -> Dict[str, Any]:
        """
        Get context as a dictionary with structured metadata.
        
        Args:
            query: The search query
        
        Returns:
            Dictionary with 'context', 'sources', and 'scores' keys
        """
        docs_with_scores = self.retrieve_with_scores(query)
        
        context_list = []
        sources = []
        scores = []
        
        for doc, score in docs_with_scores:
            context_list.append(doc.page_content)
            sources.append(self._format_source(doc))
            scores.append(float(score))
        
        return {
            "context": "\n\n".join(context_list),
            "sources": sources,
            "scores": scores
        }


def create_retriever(
    vectorstore,
    search_threshold: float = 0.3,
    top_k: int = 3
) -> EnhancedRetriever:
    """
    Create an enhanced retriever from a vector store.
    
    Args:
        vectorstore: FAISS vector store instance
        search_threshold: Minimum similarity score threshold
        top_k: Maximum number of documents to retrieve
    
    Returns:
        EnhancedRetriever instance
    """
    return EnhancedRetriever(
        vectorstore=vectorstore,
        search_threshold=search_threshold,
        top_k=top_k
    )
