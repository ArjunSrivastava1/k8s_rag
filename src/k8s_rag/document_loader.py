"""
Document loading and cleaning utilities for Kubernetes documentation.
"""

import os
import re
from typing import List, Optional
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.documents import Document


class DocumentLoader:
    """
    Handles downloading and loading Kubernetes documentation.
    
    Supports:
    - Downloading documentation from Kubernetes.io
    - Cleaning noisy files (navigation, home pages)
    - Loading local documentation files
    """
    
    # Kubernetes documentation URLs
    K8S_DOCS_BASE = "https://kubernetes.io/docs"
    
    # Known documentation pages to download
    DOC_PAGES = [
        # Core Concepts
        ("concepts/overview/", "What is Kubernetes and why is it important?"),
        ("concepts/workloads/pods/pod-life-cycle/", "What is a Kubernetes Pod and what are its key characteristics?"),
        ("concepts/workloads/controllers/deployment/", "Explain the difference between a Deployment and a StatefulSet."),
        ("concepts/workloads/controllers/statefulset/"),
        
        # Scaling
        ("concepts/workloads/controllers/deployment-scaling/", "How do I scale a deployment in Kubernetes?"),
        ("concepts/workloads/controllers/horizontal-pod-autoscaler/", "What is a HorizontalPodAutoscaler?"),
        
        # Configuration
        ("tasks/configure-pod-container/configure-pod-configmap/", "What is a Kubernetes ConfigMap and how can pods consume it?"),
        ("tasks/configure-pod-container/configure-persistent-volume-storage/", "How do I create a persistent volume?"),
        
        # Networking
        ("concepts/networking/service-types/", "Explain the different types of Kubernetes Services."),
        ("tasks/access-cluster/#accessing-the-cluster-from-outside-kubernetes", "How do I expose a service to the internet?"),
        
        # Storage
        ("concepts/storage/persistent-volumes/", "How do PersistentVolumes and PersistentVolumeClaims work?"),
        ("concepts/storage/volumes/", "What types of storage volumes are available in Kubernetes?"),
        
        # Troubleshooting
        ("troubleshooting/debugging/#pod-is-stuck-in-crashloopbackoff", "My pod is stuck in CrashLoopBackOff. How do I debug this?"),
    ]
    
    def __init__(self, docs_dir: str = "k8s_docs"):
        """
        Initialize the document loader.
        
        Args:
            docs_dir: Directory to save downloaded documentation
        """
        self.docs_dir = docs_dir
        os.makedirs(docs_dir, exist_ok=True)
    
    def download_k8s_docs(self) -> List[Document]:
        """
        Download Kubernetes documentation from official URLs.
        
        Returns:
            List of Document objects with cleaned content
        """
        documents = []
        
        for url, title in self.DOC_PAGES:
            print(f"Downloading: {url}")
            try:
                doc = self._fetch_document(url, title)
                documents.append(doc)
                print(f"  ✅ Downloaded: {title}")
            except Exception as e:
                print(f"  ❌ Failed to download {url}: {e}")
        
        return documents
    
    def _fetch_document(self, url: str, title: str) -> Document:
        """
        Fetch a single documentation page and extract content.
        
        Args:
            url: URL to fetch
            title: Title for the document
        
        Returns:
            Document object with title and content
        
        Raises:
            Exception: If download fails
        """
        # Clean the title
        clean_title = self._clean_title(title)
        
        # Fetch the page
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract content
        content = self._extract_content(soup)
        
        # Clean the content
        content = self._clean_content(content)
        
        return Document(page_content=content, metadata={"source_url": url, "title": clean_title})
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """
        Extract main content from HTML, removing navigation and boilerplate.
        
        Args:
            soup: BeautifulSoup parsed HTML
        
        Returns:
            Cleaned HTML content
        """
        # Remove navigation
        nav_elements = soup.find_all(['nav', 'aside', 'footer', 'header'])
        for element in nav_elements:
            element.decompose()
        
        # Remove ads and other noise
        noise_classes = ['ad', 'advertisement', 'promo', 'promotion']
        for element in soup.find_all(class_=lambda x: x and any(n in x.lower() for n in noise_classes)):
            element.decompose()
        
        # Get main content
        main_content = soup.find(['main', 'article', 'div', 'section'])
        if main_content:
            content = str(main_content)
        else:
            content = str(soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'pre', 'code', 'blockquote']))
        
        return content
    
    def _clean_content(self, content: str) -> str:
        """
        Clean extracted content by removing HTML tags, ads, and formatting issues.
        
        Args:
            content: Raw content string
        
        Returns:
            Cleaned text content
        """
        # Remove HTML tags but keep content
        content = re.sub(r'<[^>]+>', '', content)
        
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Remove trailing newlines
        content = content.rstrip()
        
        # Remove short lines (likely remnants of HTML)
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and len(line) > 20:  # Minimum line length
                cleaned_lines.append(line)
        
        content = '\n'.join(cleaned_lines)
        
        return content
    
    def _clean_title(self, title: str) -> str:
        """
        Clean and normalize document title.
        
        Args:
            title: Original title
        
        Returns:
            Cleaned title
        """
        # Remove special characters and normalize
        clean_title = re.sub(r'[<>"\s]+', ' ', title).strip()
        clean_title = re.sub(r'\.', '', clean_title)
        
        return clean_title
    
    def load_local_docs(self, docs_dir: str) -> List[Document]:
        """
        Load documentation from local directory.
        
        Args:
            docs_dir: Directory containing documentation files
        
        Returns:
            List of Document objects
        """
        loader = DirectoryLoader(docs_dir, glob="*.md,*.txt,*.html")
        documents = loader.load()
        
        # Clean loaded documents
        cleaned_docs = []
        for doc in documents:
            cleaned_content = self._clean_content(doc.page_content)
            if cleaned_content:
                cleaned_docs.append(Document(
                    page_content=cleaned_content,
                    metadata={"source_file": os.path.basename(doc.metadata.get("source_file", "unknown"))}
                ))
        
        return cleaned_docs
    
    def get_files_to_exclude(self) -> List[str]:
        """
        Get list of file patterns to exclude (navigation, home pages, etc.).
        
        Returns:
            List of file patterns to exclude
        """
        return [
            "_docs_.html",
            "_docs_home_.html",
            "index.html",
            "404.html",
            "search.html",
        ]


def fetch_real_k8s_docs(
    output_dir: str = "k8s_docs",
    doc_urls: Optional[List[tuple]] = None
) -> List[Document]:
    """
    Download actual documentation pages with real content.
    
    Args:
        output_dir: Directory to save downloaded files
        doc_urls: Optional list of (url, title) tuples. If None, uses default K8s docs.
    
    Returns:
        List of Document objects
    """
    loader = DocumentLoader(docs_dir=output_dir)
    
    if doc_urls is None:
        doc_urls = loader.DOC_PAGES
    
    documents = loader.download_k8s_docs()
    
    # Save documents to files
    loader._save_documents(documents)
    
    return documents


def clean_noisy_files(docs_dir: str = "k8s_docs") -> int:
    """
    Remove noisy files from documentation directory.
    
    Args:
        docs_dir: Directory to clean
    
    Returns:
        Number of files removed
    """
    removed_count = 0
    excluded = DocumentLoader().get_files_to_exclude()
    
    for filename in os.listdir(docs_dir):
        if any(filename.endswith(pattern) for pattern in excluded):
            filepath = os.path.join(docs_dir, filename)
            if os.path.isfile(filepath):
                os.remove(filepath)
                print(f"Removed {filename}")
                removed_count += 1
    
    return removed_count


def _save_documents(documents: List[Document], output_dir: str = "k8s_docs") -> None:
    """
    Save documents to individual files.
    
    Args:
        documents: List of Document objects
        output_dir: Directory to save files to
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for doc in documents:
        filename = f"{doc.metadata.get('title', 'untitled')}.md"
        filename = re.sub(r'[<>:"|?*]', '_', filename)
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# {doc.metadata.get('title', 'Untitled')}\n\n")
            f.write(doc.page_content)
