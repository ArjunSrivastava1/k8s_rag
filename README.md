<h1>
  <img src="https://raw.githubusercontent.com/ArjunSrivastava1/k8s-rag/main/assets/icon.svg" alt="k8s-rag" width="100">
</h1>

<h4>A production-ready RAG (Retrieval-Augmented Generation) system for Kubernetes documentation — semantic search, LLM-powered answers, and FastAPI deployment</h4>

<p>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white" alt="Python Version"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
  <a href="https://github.com/ArjunSrivastava1/k8s-rag/actions"><img src="https://github.com/ArjunSrivastava1/k8s-rag/workflows/CI/badge.svg" alt="CI Status"></a>
  <a href="https://huggingface.co/"><img src="https://img.shields.io/badge/🤗-HuggingFace-FFD21E" alt="HuggingFace"></a>
</p>

<p>
  <a href="#-about">About</a> •
  <a href="#-features">Features</a> •
  <a href="#-research-foundation">Research</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-api-usage">API Usage</a> •
  <a href="#-evaluation">Evaluation</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/ArjunSrivastava1/k8s-rag/main/assets/demo.gif" alt="Demo" width="600">
</p>

## 🎯 About

**K8s RAG** is a production-grade Retrieval-Augmented Generation system that makes Kubernetes documentation searchable through natural language. Built on research from Meta, Google, and Anthropic, it combines:

- **Dense retrieval** (FAISS + Sentence-BERT) for finding relevant docs
- **LLM generation** for accurate, contextual answers
- **FastAPI deployment** for real-world API access
- **RAGAS evaluation** for quality measurement

Think of it as your **Kubernetes expert assistant**—ask any question, get accurate answers grounded in official documentation.

## ✨ Features

| Category | Features |
|----------|----------|
| **🔍 Semantic Search** | FAISS vector store • Sentence-BERT embeddings • Sub-second retrieval • Top-k document selection |
| **🤖 RAG Pipeline** | LangChain orchestration • Context-aware generation • Source citation • Hallucination reduction |
| **⚡ API Layer** | FastAPI endpoints • `/query` for questions • `/health` for monitoring • JSON responses |
| **📊 Evaluation** | RAGAS metrics • Faithfulness scoring • Answer relevance • Context relevance |
| **🧠 Research-Backed** | Implements RAG, DPR, ReAct papers • MCP-ready • Agentic foundations |

## 📚 Research Foundation

This project implements concepts from foundational papers:

| Paper | Authors | Implementation |
|-------|---------|----------------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Vaswani et al. 2017 | Transformer foundation |
| [RAG: Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) | Lewis et al. 2020 | Core RAG architecture |
| [DPR: Dense Passage Retrieval](https://arxiv.org/abs/2004.04906) | Karpukhin et al. 2020 | Dual-encoder retrieval |
| [FAISS](https://arxiv.org/abs/1702.08734) | Johnson et al. 2017 | Billion-scale similarity search |
| [Sentence-BERT](https://arxiv.org/abs/1908.10084) | Reimers et al. 2019 | Embedding generation |
| [ReAct](https://arxiv.org/abs/2210.03629) | Yao et al. 2022 | Agentic reasoning |
| [MCP](https://www.anthropic.com/news/model-context-protocol) | Anthropic 2024 | Tool orchestration |

## 🚀 Quick Start

### 📦 Installation

✨Coming soon✨: the project stands as completed however a stable version and refactoring for files, secrets and requirements needs to be performed
