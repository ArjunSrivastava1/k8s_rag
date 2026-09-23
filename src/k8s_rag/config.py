"""
Configuration settings for K8s RAG.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """
    Central configuration for K8s RAG system.
    
    Attributes:
        docs_dir: Directory to download Kubernetes documentation
        data_dir: Directory to store processed data and indices
        chunk_size: Size of text chunks for embedding (characters)
        chunk_overlap: Overlap between chunks (characters)
        embeddings_model: Name of the embedding model to use
        llm_model: Name of the LLM to use
        repetition_penalty: Penalty to prevent repetition in LLM output
        retrieval_k: Number of documents to retrieve
        retrieval_threshold: Minimum similarity score threshold
        output_format: Format for output (markdown, plain, etc.)
    """
    
    # Directories
    docs_dir: str = field(default_factory=lambda: "k8s_docs")
    data_dir: str = field(default_factory=lambda: "k8s_rag_data")
    
    # Chunking parameters
    chunk_size: int = 800
    chunk_overlap: int = 100
    
    # Embedding model
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings_device: str = "cpu"  # Options: "cpu", "cuda", "mps"
    
    # LLM configuration
    llm_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    repetition_penalty: float = 1.2
    temperature: float = 0.7
    max_tokens: int = 512
    
    # Retrieval parameters
    retrieval_k: int = 3
    retrieval_threshold: float = 0.3
    
    # Output format
    output_format: str = "markdown"  # Options: "markdown", "plain", "json"
    
    # Document filtering
    exclude_patterns: list[str] = field(
        default_factory=lambda: [
            "_docs_.html",
            "_docs_home_.html",
            "index.html",
            "404.html",
        ]
    )
    
    # API configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Colab-specific
    colab_mount_point: str = "/content/drive"
    
    # Class methods for convenience
    def get_full_data_path(self, sub_dir: str) -> str:
        """Get full path for a subdirectory in the data directory."""
        return os.path.join(self.data_dir, sub_dir)
    
    def get_full_docs_path(self, sub_dir: str) -> str:
        """Get full path for a subdirectory in the docs directory."""
        return os.path.join(self.docs_dir, sub_dir)
    
    def save(self, filepath: Optional[str] = None) -> str:
        """Save configuration to a YAML file."""
        import yaml
        
        if filepath is None:
            filepath = os.path.join(self.data_dir, "config.yaml")
        
        config_dict = {
            "docs_dir": self.docs_dir,
            "data_dir": self.data_dir,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embeddings_model": self.embeddings_model,
            "embeddings_device": self.embeddings_device,
            "llm_model": self.llm_model,
            "repetition_penalty": self.repetition_penalty,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "retrieval_k": self.retrieval_k,
            "retrieval_threshold": self.retrieval_threshold,
            "output_format": self.output_format,
            "exclude_patterns": self.exclude_patterns,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "colab_mount_point": self.colab_mount_point,
        }
        
        with open(filepath, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Optional[str] = None) -> "Config":
        """Load configuration from a YAML file."""
        import yaml
        
        if filepath is None:
            filepath = os.path.join("k8s_rag_data", "config.yaml")
        
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                config_dict = yaml.safe_load(f)
                return cls(**config_dict)
        else:
            return cls()
