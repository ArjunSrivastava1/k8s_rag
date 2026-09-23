"""
Main entry point for K8s RAG.

This module provides the main orchestration logic for the K8s RAG system,
including:
- Document loading and ingestion
- Vector store creation
- LLM initialization
- RAG chain construction
- Evaluation
- API server
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from .config import Config
from .document_loader import DocumentLoader, fetch_real_k8s_docs, clean_noisy_files
from .embeddings import EmbeddingManager
from .vector_store import VectorStoreManager
from .retriever import EnhancedRetriever
from .llm import LLMManager
from .rag_chain import RAGChain
from .evaluation import EvaluationSuite


def setup_environment(config: Config = None):
    """
    Set up the environment based on configuration.
    
    Args:
        config: Config instance (uses default if None)
    
    Returns:
        Initialized config instance
    """
    if config is None:
        config = Config()
    
    # Create directories
    os.makedirs(config.docs_dir, exist_ok=True)
    os.makedirs(config.data_dir, exist_ok=True)
    
    print(f"✅ Environment setup complete!")
    print(f"   Documentation directory: {config.docs_dir}")
    print(f"   Data directory: {config.data_dir}")
    
    return config


def load_documents(config: Config) -> tuple:
    """
    Download and load Kubernetes documentation.
    
    Args:
        config: Configuration instance
    
    Returns:
        Tuple of (documents, loader)
    """
    print("\n" + "=" * 60)
    print("Step 1: Downloading Kubernetes Documentation")
    print("=" * 60)
    
    loader = DocumentLoader(docs_dir=config.docs_dir)
    
    # Download documentation
    documents = loader.download_k8s_docs()
    print(f"\n✅ Downloaded {len(documents)} documentation pages")
    
    # Remove noisy files
    removed = clean_noisy_files(config.docs_dir)
    print(f"✅ Removed {removed} noisy files")
    
    # Load remaining files
    remaining_docs = loader.load_local_docs(config.docs_dir)
    print(f"✅ Loaded {len(remaining_docs)} additional documents from local files")
    
    all_documents = documents + remaining_docs
    print(f"\n📄 Total documents loaded: {len(all_documents)}")
    
    return all_documents, loader


def ingest_documents(
    documents,
    config: Config,
    index_path: str = None
) -> VectorStoreManager:
    """
    Ingest documents into vector store.
    
    Args:
        documents: List of Document objects
        config: Configuration instance
        index_path: Path to save the index
    
    Returns:
        VectorStoreManager instance
    """
    print("\n" + "=" * 60)
    print("Step 2: Ingesting Documents")
    print("=" * 60)
    
    # Initialize embeddings
    embeddings = EmbeddingManager(
        model_name=config.embeddings_model,
        device=config.embeddings_device,
        cache_dir=config.get_full_data_path("embeddings")
    )
    
    # Initialize vector store manager
    index_path = index_path or config.get_full_data_path("k8s_faiss_index_real")
    vectorstore = VectorStoreManager(
        index_path=index_path,
        embeddings=embeddings
    )
    
    # Create vector store
    print(f"\nCreating vector store with {len(documents)} documents...")
    vectorstore.create_from_documents(
        documents=documents,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap
    )
    
    print(f"\n✅ Vector store created with {vectorstore.document_count} chunks")
    
    return vectorstore


def initialize_llm(config: Config) -> LLMManager:
    """
    Initialize the LLM.
    
    Args:
        config: Configuration instance
    
    Returns:
        LLMManager instance
    """
    print("\n" + "=" * 60)
    print("Step 3: Loading LLM")
    print("=" * 60)
    
    llm = LLMManager(
        model_name=config.llm_model,
        repetition_penalty=config.repetition_penalty,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        device=config.embeddings_device
    )
    
    print(f"\n✅ LLM initialized: {config.llm_model}")
    
    return llm


def create_rag_chain(
    retriever: EnhancedRetriever,
    llm: LLMManager
) -> RAGChain:
    """
    Create the RAG chain.
    
    Args:
        retriever: EnhancedRetriever instance
        llm: LLMManager instance
    
    Returns:
        RAGChain instance
    """
    print("\n" + "=" * 60)
    print("Step 4: Creating RAG Chain")
    print("=" * 60)
    
    rag_chain = RAGChain(
        retriever=retriever,
        llm=llm.llm,
        output_format=config.output_format
    )
    
    print(f"\n✅ RAG chain created")
    
    return rag_chain


def run_evaluation(rag_chain: RAGChain, config: Config):
    """
    Run evaluation tests.
    
    Args:
        rag_chain: RAGChain instance
        config: Configuration instance
    
    Returns:
        EvaluationSuite instance
    """
    print("\n" + "=" * 60)
    print("Step 5: Running Evaluation")
    print("=" * 60)
    
    evaluator = EvaluationSuite()
    evaluator.evaluate(rag_chain)
    evaluator.print_report()
    
    return evaluator


def create_api(
    rag_chain: RAGChain,
    vectorstore: VectorStoreManager,
    host: str = None,
    port: int = None
):
    """
    Create the API server.
    
    Args:
        rag_chain: RAGChain instance
        vectorstore: VectorStoreManager instance
        host: Host to bind (uses config if None)
        port: Port to bind (uses config if None)
    
    Returns:
        K8sRAGAPI instance
    """
    from .api import create_api as create_api_impl
    
    host = host or config.api_host
    port = port or config.api_port
    
    api = create_api_impl(
        rag_chain=rag_chain,
        vectorstore=vectorstore,
        host=host,
        port=port
    )
    
    print("\n" + "=" * 60)
    print(f"Step 6: API Server Ready")
    print("=" * 60)
    print(f"\nAPI endpoints:")
    print(f"   Health:     http://{host}:{port}/health")
    print(f"   Query:      http://{host}:{port}/query")
    print(f"   Metrics:    http://{host}:{port}/metrics")
    print(f"   Docs:       http://{host}:{port}/docs")
    print(f"\nRun: uvicorn api:app --host {host} --port {port}")
    
    return api


def main():
    """Main entry point."""
    print("=" * 60)
    print("K8s RAG - Kubernetes Documentation RAG System")
    print("=" * 60)
    
    # Load configuration
    config = Config()
    
    try:
        # Step 1: Setup environment
        config = setup_environment(config)
        
        # Step 2: Load documents
        documents, loader = load_documents(config)
        
        # Step 3: Ingest documents
        vectorstore = ingest_documents(documents, config)
        
        # Step 4: Initialize LLM
        llm = initialize_llm(config)
        
        # Step 5: Create retriever and RAG chain
        retriever = EnhancedRetriever(
            vectorstore=vectorstore,
            search_threshold=config.retrieval_threshold,
            top_k=config.retrieval_k
        )
        
        rag_chain = create_rag_chain(retriever, llm)
        
        # Step 6: Run evaluation
        evaluator = run_evaluation(rag_chain, config)
        
        # Step 7: Create API (optional)
        create_api(rag_chain, vectorstore)
        
        print("\n" + "=" * 60)
        print("🎉 K8s RAG Setup Complete!")
        print("=" * 60)
        print(f"\nYou can now:")
        print(f"   - Query the system: rag_chain.invoke('your question')")
        print(f"   - Run the API: python -m uvicorn src.k8s_rag.api:app --reload")
        
        return {
            "config": config,
            "vectorstore": vectorstore,
            "llm": llm,
            "retriever": retriever,
            "rag_chain": rag_chain,
            "evaluator": evaluator,
        }
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
