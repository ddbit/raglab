import os
import argparse
from pathlib import Path
from typing import List, Optional

from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader
from rag import RAGSystem

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

def chunk_document(text: str, metadata: dict, chunk_size: int = 1024, chunk_overlap: int = 20) -> List[tuple]:
    """Split document into chunks with metadata"""
    parser = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = parser.split_text(text)
    return [(chunk, metadata) for chunk in chunks]

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into RAG system")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing documents to ingest")
    parser.add_argument("--model", type=str, default="phi3", help="Model name to use")
    parser.add_argument("--engine", type=str, default="ollama", choices=["ollama", "openai"],
                       help="LLM engine to use (ollama or openai)")
    parser.add_argument("--api-key", type=str, help="API key for OpenAI (if using OpenAI engine)")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Document chunk size")
    parser.add_argument("--collection", type=str, default="my_documents", help="ChromaDB collection name")
    parser.add_argument("--file-type", type=str, choices=["txt", "pdf", "all"], default="all", 
                       help="Type of files to ingest: txt, pdf, or all")
    args = parser.parse_args()
    
    # Initialize RAG system
    rag = RAGSystem(
        collection_name=args.collection,
        model_name=args.model,
        engine=args.engine,
        api_key=args.api_key
    )
    
    # Read documents based on file type
    documents = []
    if args.file_type in ["txt", "all"]:
        txt_docs = read_text_files(args.dir)
        documents.extend(txt_docs)
        print(f"Found {len(txt_docs)} text documents")
        
    if args.file_type in ["pdf", "all"]:
        pdf_docs = read_pdf_files(args.dir)
        documents.extend(pdf_docs)
        print(f"Found {len(pdf_docs)} PDF document pages")
    
    if not documents:
        print(f"No {args.file_type} files found in {args.dir}")
        return
    
    print(f"Processing {len(documents)} total document chunks")
    
    # Process and add all documents
    total_chunks = 0
    for doc_text, doc_metadata in documents:
        # Split into chunks
        chunks = chunk_document(doc_text, doc_metadata, chunk_size=args.chunk_size)
        
        # Extract texts and metadatas
        texts = [chunk for chunk, _ in chunks]
        metadatas = [metadata for _, metadata in chunks]
        
        # Add to index
        added = rag.add_documents(texts, metadatas)
        total_chunks += added
        
        source_info = f"{doc_metadata['filename']}"
        if doc_metadata.get('page'):
            source_info += f" (page {doc_metadata['page']})"
        
        print(f"Added {added} chunks from {source_info}")
    
    print(f"Successfully ingested {len(documents)} document chunks with {total_chunks} total indexed chunks")

if __name__ == "__main__":
    main()