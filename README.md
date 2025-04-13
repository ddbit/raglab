# Simple RAG System with Ollama

A minimal Retrieval-Augmented Generation (RAG) implementation using LlamaIndex, ChromaDB and Ollama for local LLMs.

## Features

- Uses LlamaIndex for document indexing and querying
- ChromaDB as a lightweight local vector database
- Sentence-Transformers for embedding generation
- Integrates with Ollama for local LLM inference
- Simple API for adding documents and querying
- Support for PDF and text file ingestion

## Prerequisites

- [Ollama](https://ollama.ai/) installed and running locally
- Python 3.8+

## Setup

1. Install Ollama and pull a model (if not already done):
```bash
# Install Ollama following instructions at https://ollama.ai/
# Then pull a model:
ollama pull phi3
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Building the Knowledge Base

To build the knowledge base, place your documents in the `data` directory (and its subdirectories), then run:

```bash
python src/build.py
```

The build script will:
1. Discover all documents in the data directory and its subdirectories
2. Process each document according to its type (PDF, text, etc.)
3. Create chunks with the configured parameters
4. Add the chunks to the vector database

Additional options (all have defaults in hyperparams.py):
- `--dir`: Specify the knowledge base directory (default: "./data")
- `--model`: Specify the model name to use
- `--engine`: Choose the LLM engine ("ollama" or "openai")
- `--api-key`: OpenAI API key (required only when using OpenAI engine)
- `--temperature`: Set the temperature for generation
- `--chunk-size`: Set the document chunk size
- `--chunk-overlap`: Set the overlap between chunks
- `--collection`: Set the ChromaDB collection name
- `--file-type`: Choose file types to process ("txt", "pdf", or "all")

This is separate from the query interface - build the knowledge base first, then query it.

### Querying the System

Run the example application:
```bash
python src/app.py --model phi3 --engine ollama
```

You can use either Ollama or OpenAI as the LLM engine and customize retrieval parameters:
```bash
# Using Ollama with custom parameters
python src/app.py --engine ollama --model phi3 --temperature 0.2 --top-k 5 --retriever hybrid

# Using OpenAI with custom parameters
python src/app.py --engine openai --model gpt-3.5-turbo --api-key your_api_key --temperature 0.1 --top-k 3 --retriever dense
# or with API key in environment variable:
export OPENAI_API_KEY=your_api_key
python src/app.py --engine openai --model gpt-3.5-turbo --temperature 0.1 --top-k 3 --retriever sparse
```

Or use the interactive terminal with conversation history:
```bash
python src/query.py --interactive --engine ollama --model phi3 --temperature 0.1 --top-k 3 --retriever default
```

The interactive terminal supports these commands:
- `/list` - List all documents currently indexed in the vector database
- `/model` - Show information about the current model
- `/models` - List available models in Ollama (if using Ollama engine)
- `/switch <model>` - Switch to a different model
- `/inject <path>` - Add a single file to the vector database
- `/clear` - Clear conversation history
- `/reset` - Reset conversation context without clearing history
- `/bye` - Exit the program
- `/help` - Show all available commands

### Using the RAG system in your code

```python
from src.rag import RAGSystem
import hyperparams

# Using Ollama with default hyperparameters
rag = RAGSystem(
    collection_name=hyperparams.DEFAULT_COLLECTION_NAME,
    model_name=hyperparams.DEFAULT_MODEL_NAME,
    engine=hyperparams.DEFAULT_ENGINE,
    temperature=hyperparams.DEFAULT_TEMPERATURE
)

# Or using OpenAI with custom parameters
# rag = RAGSystem(
#     collection_name="my_documents",
#     model_name="gpt-3.5-turbo",
#     engine="openai",
#     api_key="your_api_key",  # Or set OPENAI_API_KEY environment variable
#     temperature=0.2
# )

# Add documents with custom chunking
from src.ingest import chunk_document

# Example document
doc_text = "Your document text here"
doc_metadata = {"source": "example.txt", "filename": "example.txt"}

# Create chunks with custom size
chunks = chunk_document(
    doc_text, 
    doc_metadata, 
    chunk_size=hyperparams.CHUNK_SIZE,
    chunk_overlap=hyperparams.CHUNK_OVERLAP
)

# Extract texts and metadatas
texts = [chunk for chunk, _ in chunks]
metadatas = [metadata for _, metadata in chunks]

# Add chunks to index
rag.add_documents(texts, metadatas)

# Or simply add texts directly
more_texts = ["Another document", "Yet another document"]
rag.add_documents(more_texts)

# Query with custom retrieval parameters
result = rag.query(
    "Your question here?", 
    similarity_top_k=hyperparams.TOP_K,
    retriever_type=hyperparams.RETRIEVER_TYPE
)
print(result["answer"])
```

### Resetting the Vector Database

If you need to clear all documents from the vector database:

```bash
python src/reset_db.py --collection my_documents

# To skip confirmation prompt:
python src/reset_db.py --collection my_documents --force
```

## Directory Structure

- `src/`: Source code
  - `app.py`: Main application for querying
  - `build.py`: Knowledge base builder script
  - `rag.py`: Core RAG system implementation
  - `query.py`: Query handling logic
  - `reset_db.py`: Script to reset the vector database
  - `test_rag.py`: Test script for the RAG system
  - `hyperparams.py`: Centralized hyperparameters for the RAG system
- `data/`: Directory containing knowledge base files (with subdirectories for different domains)
- `chroma_db/`: Default location for the ChromaDB vector database

## Customization

The system is highly configurable through the `hyperparams.py` file, which centralizes all adjustable parameters:

### Hyperparameters

The `src/hyperparams.py` file contains all the configurable parameters for the RAG system:

```python
# Document chunking parameters
CHUNK_SIZE = 1024  # Size of each document chunk in characters/tokens
CHUNK_OVERLAP = 20  # Overlap between chunks in characters/tokens

# Retrieval parameters
TOP_K = 3  # Number of chunks to retrieve for each query
RETRIEVER_TYPE = "default"  # "default", "sparse", "dense" or "hybrid"

# LLM parameters
DEFAULT_MODEL_NAME = "phi3"  # Default model name
DEFAULT_ENGINE = "ollama"  # Default engine (ollama or openai)
DEFAULT_TEMPERATURE = 0.1  # Temperature for generation (0.0 to 1.0)
DEFAULT_MAX_TOKENS = 1024  # Maximum number of tokens to generate

# Embedding parameters
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Model for embeddings

# Storage parameters
DEFAULT_COLLECTION_NAME = "my_documents"  # Default ChromaDB collection name
DEFAULT_PERSIST_DIR = "./chroma_db"  # Default directory to persist ChromaDB
```

You can modify these parameters to optimize the performance of your RAG system:

- Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` to find the optimal chunk size for your documents
- Experiment with different `RETRIEVER_TYPE` values to find the best retrieval method for your use case
- Tune `TOP_K` to control how many chunks are retrieved for each query
- Adjust `DEFAULT_TEMPERATURE` to control the creativity/randomness of the generated responses

All command-line scripts respect these parameters as defaults, but you can also override them using command-line arguments.

## Troubleshooting

- Make sure Ollama is running before using the system
- If you encounter errors with a specific model, try using a different Ollama model
- For large documents, consider chunking them before adding to the system