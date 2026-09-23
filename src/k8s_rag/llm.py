"""
LLM management for K8s RAG with TinyLlama.
"""

import os
import torch
from typing import Optional, Dict, Any
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


class LLMManager:
    """
    Manages local LLM integration for K8s RAG.
    
    Provides:
    - Loading TinyLlama 1.1B model
    - Repetition penalty optimization
    - Chat prompt template
    - RAG chain construction
    """
    
    # Default TinyLlama model
    DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        repetition_penalty: float = 1.2,
        temperature: float = 0.7,
        max_tokens: int = 512,
        device: str = "cpu"
    ):
        """
        Initialize the LLM manager.
        
        Args:
            model_name: Name of the HuggingFace model
            repetition_penalty: Penalty to prevent repetition
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            device: Device to use ("cpu", "cuda")
        """
        self.model_name = model_name
        self.repetition_penalty = repetition_penalty
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device = device
        
        self.llm = None
        self.pipeline = None
        self.rag_chain = None
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the LLM model and create pipeline."""
        try:
            from langchain_huggingface import HuggingFacePipeline
            
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            
            print(f"Loading LLM: {self.model_name}...")
            print("This will take 1-2 minutes to download on first run...")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.device,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=self.max_tokens,
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.9
            )
            
            self.llm = HuggingFacePipeline.from_model_id(
                model_id=self.model_name,
                model=self.pipeline,
                task="text-generation",
                model_kwargs={
                    "repetition_penalty": self.repetition_penalty,
                    "temperature": self.temperature,
                    "max_new_tokens": self.max_tokens,
                }
            )
            
            print(f"✅ LLM loaded successfully: {self.model_name}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load LLM: {e}")
    
    def create_rag_chain(
        self,
        retriever,
        prompt_template: Optional[str] = None
    ) -> Any:
        """
        Create a RAG chain using the LLM and retriever.
        
        Args:
            retriever: Retriever instance
            prompt_template: Optional custom prompt template
        
        Returns:
            RAG chain instance
        """
        from langchain_core.prompts import ChatPromptTemplate
        
        # Default prompt for Kubernetes Q&A
        default_prompt = """You are a Kubernetes expert assistant. Answer the question based ONLY on the context provided below.

If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question" rather than making things up.

When providing commands, ensure they are accurate and copy-paste ready.

Context:
{context}

Question: {question}

Answer:"""
        
        if prompt_template is None:
            prompt = ChatPromptTemplate.from_template(default_prompt)
        else:
            prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Create the RAG chain using LCEL
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        self.rag_chain = rag_chain
        
        return rag_chain
    
    def format_docs(self, documents) -> str:
        """
        Format documents for the RAG prompt.
        
        Args:
            documents: List of Document objects
        
        Returns:
            Formatted context string
        """
        document_lines = []
        for doc in documents:
            source_info = self._format_source(doc)
            content = doc.page_content
            document_lines.append(f"{source_info}\n{content}")
        
        return "\n\n".join(document_lines)
    
    def _format_source(self, doc) -> str:
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
        
        if source_url:
            from urllib.parse import urlparse
            parsed = urlparse(source_url)
            path = parsed.path
            source = path.rstrip('/').split('/')[-1] if path else "Kubernetes Documentation"
        elif source_file:
            source = source_file
        else:
            source = "Kubernetes Documentation"
        
        return f"[Source: {source}]"
    
    def invoke(self, query: str) -> str:
        """
        Invoke the RAG chain with a query.
        
        Args:
            query: The question to answer
        
        Returns:
            Generated answer
        """
        if self.rag_chain is None:
            raise ValueError("RAG chain not initialized. Create it with create_rag_chain().")
        
        return self.rag_chain.invoke(query)
    
    @property
    def is_loaded(self) -> bool:
        """Check if the LLM is loaded."""
        return self.llm is not None


def create_rag_chain(
    retriever,
    llm,
    prompt_template: Optional[str] = None
) -> Any:
    """
    Create a RAG chain from a retriever and LLM.
    
    Args:
        retriever: Retriever instance
        llm: LLM instance
        prompt_template: Optional custom prompt template
    
    Returns:
        RAG chain instance
    """
    manager = LLMManager()
    manager.llm = llm
    return manager.create_rag_chain(retriever, prompt_template)


def create_prompt_template() -> ChatPromptTemplate:
    """
    Create the default Kubernetes expert prompt.
    
    Returns:
        ChatPromptTemplate instance
    """
    return ChatPromptTemplate.from_template("""You are a Kubernetes expert assistant. Answer the question based ONLY on the context provided below.

If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question" rather than making things up.

When providing commands, ensure they are accurate and copy-paste ready.

Context:
{context}

Question: {question}

Answer:""")
