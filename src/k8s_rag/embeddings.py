"""
Embedding management for K8s RAG.
"""

import os
from typing import List, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings


class EmbeddingManager:
    """
    Manages embedding model initialization and caching.
    
    Provides:
    - Loading HuggingFace embedding models
    - Caching embeddings to disk for faster retrieval
    - Device selection (CPU/GPU)
    """
    
    # Default embedding model
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Cache directory for embeddings
    CACHE_DIR = "sentence_embeddings"
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        show_download_progress: bool = True
    ):
        """
        Initialize the embedding manager.
        
        Args:
            model_name: Name of the HuggingFace embedding model
            device: Device to use ("cpu", "cuda", "mps")
            cache_dir: Directory to cache embeddings
            show_download_progress: Whether to show download progress
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir or os.path.join("k8s_rag_data", self.CACHE_DIR)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.embeddings: Optional[Embeddings] = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the embedding model."""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": self.device},
                cache_folder=self.cache_dir,
                show_download_progress=self.show_download_progress
            )
            print(f"✅ Loaded embedding model: {self.model_name} on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            List of embedding vectors
        """
        if self.embeddings is None:
            self._load_model()
        
        return self.embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            text: Query text to embed
        
        Returns:
            Embedding vector for the query
        """
        if self.embeddings is None:
            self._load_model()
        
        return self.embeddings.embed_query(text)
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the embedding model to disk.
        
        Args:
            filepath: Path to save the model (uses cache_dir by default)
        
        Returns:
            Path where the model was saved
        """
        if filepath is None:
            filepath = os.path.join(self.cache_dir, "embedding_model")
        
        if self.embeddings is not None:
            self.embeddings.save_local(filepath)
            print(f"✅ Saved embedding model to {filepath}")
        
        return filepath
    
    @classmethod
    def load(cls, model_name: str, cache_dir: str, device: str = "cpu") -> "EmbeddingManager":
        """
        Load a previously saved embedding model.
        
        Args:
            model_name: Name of the embedding model
            cache_dir: Directory where the model is cached
            device: Device to use
        
        Returns:
            Loaded EmbeddingManager instance
        """
        manager = cls.__new__(cls)
        manager.model_name = model_name
        manager.device = device
        manager.cache_dir = cache_dir
        manager.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            cache_folder=cache_dir
        )
        return manager


def create_embeddings(
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: str = "cpu"
) -> List[List[float]]:
    """
    Create embeddings for a list of texts.
    
    Args:
        texts: List of text strings
        model_name: Name of the embedding model
        device: Device to use
    
    Returns:
        List of embedding vectors
    """
    manager = EmbeddingManager(model_name=model_name, device=device)
    return manager.embed_documents(texts)
