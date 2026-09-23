"""
RAG chain implementation for K8s RAG.
"""

from typing import Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document


class RAGChain:
    """
    RAG chain for Kubernetes Q&A.
    
    Provides:
    - Document retrieval and formatting
    - Prompt construction
    - Answer generation
    """
    
    DEFAULT_PROMPT = """You are a Kubernetes expert assistant. Answer the question based ONLY on the context provided below.

If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question" rather than making things up.

When providing commands, ensure they are accurate and copy-paste ready.

Context:
{context}

Question: {question}

Answer:"""
    
    def __init__(
        self,
        retriever,
        llm,
        prompt_template: Optional[str] = None,
        output_format: str = "markdown"
    ):
        """
        Initialize the RAG chain.
        
        Args:
            retriever: Retriever instance
            llm: LLM instance
            prompt_template: Optional custom prompt template
            output_format: Output format ("markdown", "plain", "json")
        """
        self.retriever = retriever
        self.llm = llm
        self.output_format = output_format
        
        # Create prompt
        if prompt_template is None:
            self.prompt = ChatPromptTemplate.from_template(self.DEFAULT_PROMPT)
        else:
            self.prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Build the chain
        self.chain = (
            {"context": self._format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _format_docs(self, documents: List[Document]) -> str:
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
            Generated answer string
        """
        return self.chain.invoke(query)
    
    def batch(self, queries: List[str]) -> List[str]:
        """
        Invoke the RAG chain with multiple queries.
        
        Args:
            queries: List of questions to answer
        
        Returns:
            List of generated answers
        """
        return self.chain.batch(queries)
    
    def with_sources(self, query: str) -> dict:
        """
        Invoke the RAG chain and return answer with sources.
        
        Args:
            query: The question to answer
        
        Returns:
            Dictionary with 'answer', 'sources', and 'context'
        """
        docs = self.retriever.retrieve(query)
        context = self._format_docs(docs)
        answer = self.chain.invoke({"context": context, "question": query})
        
        return {
            "answer": answer,
            "sources": [
                doc.metadata.get("source_url", doc.metadata.get("source_file", "Unknown"))
                for doc in docs
            ],
            "context": context
        }


def create_rag_chain(
    retriever,
    llm,
    prompt_template: Optional[str] = None,
    output_format: str = "markdown"
) -> RAGChain:
    """
    Create a RAG chain instance.
    
    Args:
        retriever: Retriever instance
        llm: LLM instance
        prompt_template: Optional custom prompt template
        output_format: Output format ("markdown", "plain", "json")
    
    Returns:
        RAGChain instance
    """
    return RAGChain(
        retriever=retriever,
        llm=llm,
        prompt_template=prompt_template,
        output_format=output_format
    )
