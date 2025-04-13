import os
import argparse
from pathlib import Path
import glob
from typing import List, Optional
import sys

from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader
from rag import RAGSystem
import hyperparams

def read_text_files(directory: str) -> List[tuple]:
    """Read all text files from directory and return (content, metadata) tuples"""
    documents = []
    for file_path in Path(directory).glob("**/*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Create metadata with file info
        metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "filetype": "text"
        }
        
        documents.append((content, metadata))
    
    return documents

def read_pdf_files(directory: str) -> List[tuple]:
    """Read all PDF files from directory and return (content, metadata) tuples"""
    documents = []
    pdf_reader = PDFReader()
    
    for file_path in Path(directory).glob("**/*.pdf"):
        # Read PDF
        try:
            pdf_docs = pdf_reader.load_data(str(file_path))
            
            for i, doc in enumerate(pdf_docs):
                # Create metadata with file info
                metadata = {
                    "source": str(file_path),
                    "filename": file_path.name,
                    "filetype": "pdf",
                    "page": i + 1
                }
                
                documents.append((doc.text, metadata))
            
            print(f"Processed {file_path.name}: {len(pdf_docs)} pages")
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")
    
    return documents

def chunk_document(text: str, metadata: dict, chunk_size: int = None, chunk_overlap: int = None) -> List[tuple]:
    """Split document into chunks with metadata"""
    # Use provided values or defaults from hyperparams
    chunk_size = chunk_size if chunk_size is not None else hyperparams.CHUNK_SIZE
    chunk_overlap = chunk_overlap if chunk_overlap is not None else hyperparams.CHUNK_OVERLAP
    
    parser = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = parser.split_text(text)
    return [(chunk, metadata) for chunk in chunks]

def process_single_file(
    file_path: str,
    rag_system = None,
    collection_name: str = None,
    model_name: str = None,
    engine: str = None, 
    api_key: str = None,
    temperature: float = None,
    chunk_size: int = None,
    chunk_overlap: int = None
):
    """Process a single file and add it to the RAG system"""
    # Use hyperparams defaults if not specified
    collection_name = collection_name or hyperparams.DEFAULT_COLLECTION_NAME
    model_name = model_name or hyperparams.DEFAULT_MODEL_NAME
    engine = engine or hyperparams.DEFAULT_ENGINE
    temperature = temperature or hyperparams.DEFAULT_TEMPERATURE
    chunk_size = chunk_size or hyperparams.CHUNK_SIZE
    chunk_overlap = chunk_overlap or hyperparams.CHUNK_OVERLAP
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' does not exist.")
        return 0
    
    # Initialize RAG system if not provided
    if rag_system is None:
        rag_system = RAGSystem(
            collection_name=collection_name,
            model_name=model_name,
            engine=engine,
            api_key=api_key,
            temperature=temperature
        )
    
    # Handle different file types
    documents = []
    if file_path.lower().endswith('.pdf'):
        pdf_reader = PDFReader()
        try:
            pdf_docs = pdf_reader.load_data(file_path)
            
            for i, doc in enumerate(pdf_docs):
                # Create metadata with file info
                metadata = {
                    "source": str(file_path),
                    "filename": os.path.basename(file_path),
                    "filetype": "pdf",
                    "page": i + 1
                }
                
                documents.append((doc.text, metadata))
            
            print(f"Processed {os.path.basename(file_path)}: {len(pdf_docs)} pages")
        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {str(e)}")
            
    elif file_path.lower().endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metadata = {
            "source": file_path,
            "filename": os.path.basename(file_path),
            "filetype": "text"
        }
        
        documents.append((content, metadata))
    else:
        print(f"Error: Unsupported file type for {file_path}. Only PDF and TXT files are supported.")
        return 0
    
    if not documents:
        print(f"Error: No content extracted from file {file_path}.")
        return 0
    
    # Process and add documents
    total_chunks = 0
    for doc_text, doc_metadata in documents:
        # Add domain from folder name
        doc_metadata["domain"] = os.path.basename(os.path.dirname(file_path))
        
        # Split into chunks
        chunks = chunk_document(doc_text, doc_metadata, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Extract texts and metadatas
        texts = [chunk for chunk, _ in chunks]
        metadatas = [metadata for _, metadata in chunks]
        
        # Add to index
        added = rag_system.add_documents(texts, metadatas)
        total_chunks += added
        
        source_info = f"{doc_metadata['filename']}"
        if doc_metadata.get('page'):
            source_info += f" (page {doc_metadata['page']})"
        
        print(f"Added {added} chunks from {source_info}")
    
    return total_chunks

def build_knowledge_base(
    directory: str = None, 
    collection_name: str = None,
    model_name: str = None,
    engine: str = None, 
    api_key: str = None,
    temperature: float = None,
    chunk_size: int = None,
    chunk_overlap: int = None,
    file_type: str = "all"
):
    """Build the knowledge base by ingesting documents from the specified directory"""
    # Use hyperparams defaults if not specified
    directory = directory or hyperparams.KNOWLEDGE_BASE_DIR
    collection_name = collection_name or hyperparams.DEFAULT_COLLECTION_NAME
    model_name = model_name or hyperparams.DEFAULT_MODEL_NAME
    engine = engine or hyperparams.DEFAULT_ENGINE
    temperature = temperature or hyperparams.DEFAULT_TEMPERATURE
    chunk_size = chunk_size or hyperparams.CHUNK_SIZE
    chunk_overlap = chunk_overlap or hyperparams.CHUNK_OVERLAP
    
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    # Initialize RAG system
    rag = RAGSystem(
        collection_name=collection_name,
        model_name=model_name,
        engine=engine,
        api_key=api_key,
        temperature=temperature
    )
    
    # Discover subdirectories - each one is a separate knowledge domain
    subfolders = [f.path for f in os.scandir(directory) if f.is_dir()]
    if not subfolders:
        subfolders = [directory]  # Use the main directory if no subfolders exist
    
    total_documents = 0
    total_chunks = 0
    
    for subfolder in subfolders:
        subfolder_name = os.path.basename(subfolder)
        print(f"\nProcessing folder: {subfolder} ({subfolder_name})")
        
        # Read documents based on file type
        documents = []
        if file_type in ["txt", "all"]:
            txt_docs = read_text_files(subfolder)
            documents.extend(txt_docs)
            print(f"Found {len(txt_docs)} text documents")
            
        if file_type in ["pdf", "all"]:
            pdf_docs = read_pdf_files(subfolder)
            documents.extend(pdf_docs)
            print(f"Found {len(pdf_docs)} PDF document pages")
        
        if not documents:
            print(f"No {file_type} files found in {subfolder}")
            continue
        
        print(f"Processing {len(documents)} documents from {subfolder_name}")
        
        # Process and add all documents
        subfolder_chunks = 0
        for doc_text, doc_metadata in documents:
            # Add subfolder as domain in metadata
            doc_metadata["domain"] = subfolder_name
            
            # Split into chunks
            chunks = chunk_document(doc_text, doc_metadata, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            # Extract texts and metadatas
            texts = [chunk for chunk, _ in chunks]
            metadatas = [metadata for _, metadata in chunks]
            
            # Add to index
            added = rag.add_documents(texts, metadatas)
            total_chunks += added
            subfolder_chunks += added
            
            source_info = f"{doc_metadata['filename']}"
            if doc_metadata.get('page'):
                source_info += f" (page {doc_metadata['page']})"
            
            print(f"Added {added} chunks from {source_info}")
        
        print(f"Successfully processed {len(documents)} documents with {subfolder_chunks} total indexed chunks from {subfolder_name}")
        total_documents += len(documents)
    
    print(f"\nKnowledge base build complete:")
    print(f"- Total documents processed: {total_documents}")
    print(f"- Total chunks indexed: {total_chunks}")
    print(f"- Collection name: {collection_name}")
    print(f"- Model: {model_name}")
    print(f"- Engine: {engine}")
    print(f"- Chunk size: {chunk_size}")
    print(f"- Chunk overlap: {chunk_overlap}")

def main():
    parser = argparse.ArgumentParser(description="Build knowledge base by ingesting documents")
    parser.add_argument("--dir", type=str, default=hyperparams.KNOWLEDGE_BASE_DIR, help="Directory containing documents to ingest")
    parser.add_argument("--model", type=str, default=hyperparams.DEFAULT_MODEL_NAME, help="Model name to use")
    parser.add_argument("--engine", type=str, default=hyperparams.DEFAULT_ENGINE, choices=["ollama", "openai"],
                       help="LLM engine to use (ollama or openai)")
    parser.add_argument("--api-key", type=str, help="API key for OpenAI (if using OpenAI engine)")
    parser.add_argument("--temperature", type=float, default=hyperparams.DEFAULT_TEMPERATURE, help="Temperature for generation")
    parser.add_argument("--chunk-size", type=int, default=hyperparams.CHUNK_SIZE, help="Document chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=hyperparams.CHUNK_OVERLAP, help="Overlap between chunks")
    parser.add_argument("--collection", type=str, default=hyperparams.DEFAULT_COLLECTION_NAME, help="ChromaDB collection name")
    parser.add_argument("--file-type", type=str, choices=["txt", "pdf", "all"], default="all", 
                       help="Type of files to ingest: txt, pdf, or all")
    args = parser.parse_args()
    
    # Build the knowledge base
    build_knowledge_base(
        directory=args.dir,
        collection_name=args.collection,
        model_name=args.model,
        engine=args.engine,
        api_key=args.api_key,
        temperature=args.temperature,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        file_type=args.file_type
    )

if __name__ == "__main__":
    main()