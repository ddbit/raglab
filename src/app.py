import os
import argparse
from rag import RAGSystem
import hyperparams

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RAG Demo Application")
    parser.add_argument("--model", type=str, default=hyperparams.DEFAULT_MODEL_NAME, help="Model name to use")
    parser.add_argument("--engine", type=str, default=hyperparams.DEFAULT_ENGINE, choices=["ollama", "openai"], 
                        help="LLM engine to use (ollama or openai)")
    parser.add_argument("--collection", type=str, default=hyperparams.DEFAULT_COLLECTION_NAME, 
                        help="ChromaDB collection name")
    parser.add_argument("--api-key", type=str, help="API key for OpenAI (if using OpenAI engine)")
    parser.add_argument("--temperature", type=float, default=hyperparams.DEFAULT_TEMPERATURE, 
                       help="Temperature for generation (0.0 to 1.0)")
    parser.add_argument("--top-k", type=int, default=hyperparams.TOP_K, 
                       help="Number of documents to retrieve")
    parser.add_argument("--retriever", type=str, default=hyperparams.RETRIEVER_TYPE, 
                       choices=["default", "sparse", "dense", "hybrid"], help="Type of retriever to use")
    args = parser.parse_args()
    
    # Initialize the RAG system with the specified model and engine
    rag = RAGSystem(
        collection_name=args.collection,
        model_name=args.model,
        engine=args.engine,
        api_key=args.api_key,
        temperature=args.temperature
    )
    
    # Example documents
    sample_texts = [
        "LlamaIndex is a data framework for LLM applications to ingest, structure, and access private or domain-specific data.",
        "ChromaDB is a database for building AI applications with embeddings.",
        "Retrieval-Augmented Generation (RAG) combines retrieval of external knowledge with text generation.",
        "Vector databases store embeddings and enable semantic search based on vector similarity.",
        "Ollama is an open-source framework for running and serving LLMs locally."
    ]
    
    # Add documents to the index
    rag.add_documents(sample_texts)
    print(f"Added {len(sample_texts)} documents to the index")
    
    # Query the RAG system
    query = "What is RAG and how does it work with local LLMs?"
    result = rag.query(
        query,
        similarity_top_k=args.top_k,
        retriever_type=args.retriever
    )
    
    print("\nQuery:", query)
    print("\nAnswer:", result["answer"])
    print("\nSources:")
    for i, source in enumerate(result["source_nodes"]):
        print(f"\n[{i+1}] {source['text']} (Score: {source['score']:.4f})")

if __name__ == "__main__":
    main()