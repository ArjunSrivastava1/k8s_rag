"""
K8s RAG - A production-ready RAG system for Kubernetes documentation.

This package provides a retrieval-augmented generation system that makes
Kubernetes documentation searchable through natural language queries with
zero hallucinations and high factual accuracy.
"""

__version__ = "0.2.0"
__author__ = "Arjun Srivastava"

from .config import Config
from .document_loader import DocumentLoader
from .embeddings import EmbeddingManager
from .vector_store import VectorStoreManager
from .retriever import Retriever
from .llm import LLMManager
from .rag_chain import RAGChain
from .evaluation import EvaluationSuite
from .api import API

__all__ = [
    "Config",
    "DocumentLoader",
    "EmbeddingManager",
    "VectorStoreManager",
    "Retriever",
    "LLMManager",
    "RAGChain",
    "EvaluationSuite",
    "API",
]
