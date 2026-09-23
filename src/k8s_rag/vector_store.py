"""
Vector store management for K8s RAG.
"""

import os
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class VectorStoreManager:
    """
    Manages FAISS vector store for document retrieval.
    
    Provides:
    - Creating vector store from documents and embeddings
    - Loading saved vector store
    - Optimized chunking for RAG
    """
    
    # Default index path
    DEFAULT_INDEX_PATH = "k8s_rag_data/k8s_faiss_index_real"
    
    def __init__(
        self,
        index_path: Optional[str] = None,
        embeddings: Optional[Embeddings] = None
    ):
        """
        Initialize the vector store manager.
        
        Args:
            index_path: Path to save/load the FAISS index
            embeddings: Optional embeddings instance (will be loaded if not provided)
        """
        self.index_path = index_path or self.DEFAULT_INDEX_PATH
        self.embeddings = embeddings
        self.vectorstore: Optional[FAISS] = None
        
        # Ensure index directory exists
        os.makedirs(self.index_path, exist_ok=True)
    
    def create_from_documents(
        self,
        documents: List[Document],
        chunk_size: int = 800,
        chunk_overlap: int = 100
    ) -> FAISS:
        """
        Create a FAISS vector store from documents.
        
        Args:
            documents: List of Document objects
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        
        Returns:
            FAISS vector store instance
        """
        if self.embeddings is None:
            raise ValueError("Embeddings instance is required. Pass it to __init__.")
        
        # Split documents into chunks
        chunks = self._split_documents(documents, chunk_size, chunk_overlap)
        
        print(f"Creating vector store with {len(chunks)} chunks...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        # Save the vector store
        self.vectorstore.save_local(self.index_path)
        print(f"✅ Saved vector store to {self.index_path}")
        
        return self.vectorstore
    
    def _split_documents(
        self,
        documents: List[Document],
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Document]:
        """
        Split documents into overlapping chunks.
        
        Args:
            documents: List of Document objects
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks
        
        Returns:
            List of chunked Document objects
        """
        chunks = []
        
        for doc in documents:
            # Split the content into chunks
            texts = self._text_splitter.split_text(doc.page_content)
            
            for text in texts:
                chunks.append(Document(
                    page_content=text,
                    metadata={
                        **doc.metadata,
                        "chunk": len(chunks),
                        "source": doc.metadata.get("source_url", doc.metadata.get("source_file", "unknown")),
                    }
                ))
        
        return chunks
    
    def _text_splitter(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Text to split
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
        
        Returns:
            List of text chunks
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
            keep_separator=False,
        )
        
        return splitter.split_text(text)
    
    def load(self) -> FAISS:
        """
        Load a saved FAISS vector store.
        
        Returns:
            Loaded FAISS vector store
        
        Raises:
            FileNotFoundError: If the index doesn't exist
        """
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Vector store index not found at {self.index_path}")
        
        print(f"Loading vector store from {self.index_path}...")
        self.vectorstore = FAISS.load_local(
            self.index_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"✅ Loaded vector store with {self.vectorstore.index.ntotal} documents")
        
        return self.vectorstore
    
    def as_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: Optional[dict] = None,
        document_count: Optional[int] = None
    ):
        """
        Create a retriever from the vector store.
        
        Args:
            search_type: Type of search ("similarity", "similarity_score", "mmr")
            search_kwargs: Additional search parameters
            document_count: Maximum number of documents to return
        
        Returns:
            LangChain retriever instance
        """
        from langchain_core.retrievers import ContextualReactorRetriever
        
        search_kwargs = search_kwargs or {}
        
        if search_type == "similarity":
            retriever = self.vectorstore.as_retriever(
                search_kwargs=search_kwargs,
                document_count=document_count
            )
        elif search_type == "similarity_score":
            retriever = self.vectorstore.as_retriever(
                search_type="similarity_score",
                search_kwargs={"k": 3, "score_threshold": 0.3}
            )
        elif search_type == "mmr":
            from langchain_retrievers import MaxMarginalRelevanceRetriever
            retriever = MaxMarginalRelevanceRetriever(
                fetch_k=document_count or 30,
                lambda_mult=0.5,
                search_kwargs=search_kwargs,
                search_kwargs_no_mmr=search_kwargs
            )
        else:
            raise ValueError(f"Unknown search type: {search_type}")
        
        return retriever
    
    @property
    def document_count(self) -> int:
        """Get the number of documents in the vector store."""
        if self.vectorstore is not None:
            return self.vectorstore.index.ntotal
        return 0
    
    def get_context(self, question: str, k: int = 3, threshold: float = 0.3) -> str:
        """
        Get relevant context for a question.
        
        Args:
            question: The question to search for
            k: Number of documents to retrieve
            threshold: Minimum similarity score threshold
        
        Returns:
            Formatted context string
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call load() or create_from_documents() first.")
        
        # Get relevant documents
        docs = self.vectorstore.similarity_search_with_score(
            question,
            k=k
        )
        
        # Filter by threshold
        filtered_docs = [
            (doc, score) for doc, score in docs
            if score >= threshold
        ]
        
        # Format context
        context_parts = []
        for doc, _ in filtered_docs:
            context_parts.append(f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}")
        
        return "\n\n".join(context_parts)


def create_vector_store(
    documents: List[Document],
    index_path: str = "k8s_rag_data/k8s_faiss_index_real",
    embeddings: Optional[Embeddings] = None,
    chunk_size: int = 800,
    chunk_overlap: int = 100
) -> FAISS:
    """
    Create a FAISS vector store from documents.
    
    Args:
        documents: List of Document objects
        index_path: Path to save the FAISS index
        embeddings: Embeddings instance
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    
    Returns:
        FAISS vector store instance
    """
    manager = VectorStoreManager(index_path=index_path, embeddings=embeddings)
    return manager.create_from_documents(
        documents=documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
