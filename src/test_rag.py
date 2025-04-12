#!/usr/bin/env python3
"""
Test script for the RAG system that:
1. Indexes a PDF file
2. Makes an inference relevant to the file content
3. Removes the file from the index
"""

import os
import sys
import argparse
import uuid
import shutil
from pathlib import Path

from rag import RAGSystem
from ingest import PDFReader, chunk_document

def test_rag_with_pdf(pdf_path, test_query, engine="ollama", model_name="phi3", api_key=None):
    """
    Run a simple test of the RAG system with a PDF file
    
    Args:
        pdf_path: Path to the PDF file to test with
        test_query: Query string to test with
        engine: LLM engine to use ('ollama' or 'openai')
        model_name: Model name to use
        api_key: API key for OpenAI (only needed if engine is 'openai')
    
    Returns:
        True if the test passed, False otherwise
    """
    print(f"\n=== Testing RAG with {pdf_path} ===\n")
    
    # Create a unique collection name for this test to isolate it
    test_collection = f"test_collection_{uuid.uuid4().hex[:8]}"
    
    # Initialize RAG system with specified engine and model
    rag = RAGSystem(
        collection_name=test_collection,
        model_name=model_name,
        engine=engine,
        api_key=api_key
    )
    
    try:
        # 1. Load and index the PDF
        print(f"Loading PDF: {pdf_path}")
        pdf_reader = PDFReader()
        
        try:
            pdf_docs = pdf_reader.load_data(pdf_path)
            
            if not pdf_docs:
                print(f"Error: No content extracted from {pdf_path}")
                return False
                
            print(f"Extracted {len(pdf_docs)} pages from PDF")
            
            # Process each page
            total_chunks = 0
            for i, doc in enumerate(pdf_docs):
                # Create metadata
                metadata = {
                    "source": pdf_path,
                    "filename": os.path.basename(pdf_path),
                    "filetype": "pdf",
                    "page": i + 1
                }
                
                # Split into chunks
                chunks = chunk_document(doc.text, metadata)
                
                # Extract texts and metadatas
                texts = [chunk for chunk, _ in chunks]
                metadatas = [metadata for _, metadata in chunks]
                
                # Add to index
                added = rag.add_documents(texts, metadatas)
                total_chunks += added
                
                print(f"Added {added} chunks from page {i+1}")
            
            print(f"Total chunks indexed: {total_chunks}")
            
            # 2. Test query
            print(f"\nTest Query: '{test_query}'")
            result = rag.query(test_query)
            
            # Print the result
            print("\nRAG Response:")
            print("="*50)
            print(result["answer"])
            print("="*50)
            
            # Print sources
            print("\nSources:")
            for i, source in enumerate(result["source_nodes"]):
                source_info = f"{source.get('metadata', {}).get('filename', 'unknown')}"
                if page := source.get('metadata', {}).get('page'):
                    source_info += f" (page {page})"
                
                print(f"\n[{i+1}] {source_info} (Score: {source['score']:.4f})")
                # Print a snippet of the text
                text = source.get('text', '')
                if text:
                    preview = text[:100] + "..." if len(text) > 100 else text
                    print(f"    {preview}")
            
            return True
            
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return False
            
    finally:
        # 3. Clean up - delete the test collection
        print(f"\nCleaning up test collection: {test_collection}")
        try:
            # Delete the collection using the ChromaDB client
            rag.chroma_client.delete_collection(test_collection)
            print("Successfully removed test collection")
        except Exception as e:
            print(f"Error removing collection: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Test the RAG system with a PDF file")
    parser.add_argument("--pdf", type=str, default="data/pdfs/cv.pdf", help="Path to the PDF file to test with")
    parser.add_argument("--query", type=str, default="Summarize this document", help="Query to test with")
    parser.add_argument("--model", type=str, default="phi4", help="Model name to use")
    parser.add_argument("--engine", type=str, default="ollama", choices=["ollama", "openai"], 
                        help="LLM engine to use (ollama or openai)")
    parser.add_argument("--api-key", type=str, help="API key for OpenAI (if using OpenAI engine)")
    
    args = parser.parse_args()
    
    # Expand the path if it's a user directory
    pdf_path = os.path.expanduser(args.pdf)
    
    # Check if the PDF exists
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' does not exist")
        sys.exit(1)
    
    # Run the test
    success = test_rag_with_pdf(
        pdf_path=pdf_path,
        test_query=args.query,
        engine=args.engine,
        model_name=args.model,
        api_key=args.api_key
    )
    
    if success:
        print("\n✅ Test completed successfully")
    else:
        print("\n❌ Test failed")
        sys.exit(1)

if __name__ == "__main__":
    main()