"""
Hyperparameters for the RAG system
This file contains all configurable parameters for the RAG system.
"""

# Document chunking parameters
CHUNK_SIZE = 1024  # Size of each document chunk in characters/tokens
CHUNK_OVERLAP = 20  # Overlap between chunks in characters/tokens

# Retrieval parameters
TOP_K = 3  # Number of chunks to retrieve for each query
RETRIEVER_TYPE = "default"  # "default", "sparse", "dense" or "hybrid"

# LLM parameters
DEFAULT_MODEL_NAME = "phi3:14b-medium-128k-instruct-fp16"  # Default model name
DEFAULT_ENGINE = "ollama"  # Default engine (ollama or openai)
DEFAULT_TEMPERATURE = 0.1  # Temperature for generation (0.0 to 1.0)
DEFAULT_MAX_TOKENS = 1024  # Maximum number of tokens to generate

# Embedding parameters
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Model for embeddings

# Storage parameters
DEFAULT_COLLECTION_NAME = "my_documents"  # Default ChromaDB collection name
DEFAULT_PERSIST_DIR = "./chroma_db"  # Default directory to persist ChromaDB

# Knowledge base parameters
KNOWLEDGE_BASE_DIR = "./data"  # Directory containing knowledge base files to ingest