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

### Adding Documents

To add documents to the system, place your PDF files in the `data/pdfs` directory, then run:

```bash
python src/ingest.py --dir data/pdfs --file-type pdf --collection my_documents
```

You can also ingest text files by using `--file-type txt` or both text and PDF files with `--file-type all`.

Additional options:
- `--model`: Specify the model name to use (default: "phi3")
- `--engine`: Choose the LLM engine ("ollama" or "openai", default: "ollama")
- `--api-key`: OpenAI API key (required only when using OpenAI engine)
- `--chunk-size`: Set the document chunk size (default: 1024)
- `--collection`: Set the ChromaDB collection name (default: "my_documents")

### Querying the System

Run the example application:
```bash
python src/app.py --model phi3 --engine ollama
```

You can use either Ollama or OpenAI as the LLM engine:
```bash
# Using Ollama
python src/app.py --engine ollama --model phi3

# Using OpenAI
python src/app.py --engine openai --model gpt-3.5-turbo --api-key your_api_key
# or with API key in environment variable:
export OPENAI_API_KEY=your_api_key
python src/app.py --engine openai --model gpt-3.5-turbo
```

Or use the interactive terminal with conversation history:
```bash
python src/query.py --interactive --engine ollama --model phi3
```

The interactive terminal supports these commands:
- `/inject <path>` - Add a file (PDF or text) to the vector database
- `/list` - List all documents currently indexed in the vector database
- `/model` - Show information about the current model
- `/models` - List available models in Ollama (if using Ollama engine)
- `/switch <model>` - Switch to a different model
- `/clear` - Clear conversation history
- `/reset` - Reset conversation context without clearing history
- `/bye` - Exit the program
- `/help` - Show all available commands

### Using the RAG system in your code

```python
from src.rag import RAGSystem

# Using Ollama
rag = RAGSystem(
    collection_name="my_documents",
    model_name="phi3",
    engine="ollama"
)

# Or using OpenAI
# rag = RAGSystem(
#     collection_name="my_documents",
#     model_name="gpt-3.5-turbo",
#     engine="openai",
#     api_key="your_api_key"  # Or set OPENAI_API_KEY environment variable
# )

# Add documents
texts = ["Your document text here", "Another document"]
rag.add_documents(texts)

# Query
result = rag.query("Your question here?")
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
  - `ingest.py`: Document ingestion script
  - `rag.py`: Core RAG system implementation
  - `query.py`: Query handling logic
  - `reset_db.py`: Script to reset the vector database
  - `test_rag.py`: Test script for the RAG system
- `data/pdfs/`: Directory for storing PDF documents to be ingested
- `chroma_db/`: Default location for the ChromaDB vector database

## Customization

- Change the Ollama model in `RAGSystem.__init__` or when initializing the class
- Modify embedding model with the `embedding_model_name` parameter
- Adjust similarity parameters in the `query` method
- Configure persistence directory with the `persist_dir` parameter

## Troubleshooting

- Make sure Ollama is running before using the system
- If you encounter errors with a specific model, try using a different Ollama model
- For large documents, consider chunking them before adding to the system