#!/usr/bin/env python3
"""
Script to reset the vector database (delete all documents from the ChromaDB collection)
"""

import os
import argparse
import chromadb
from pathlib import Path

def reset_collection(collection_name="my_documents", persist_dir="./chroma_db", confirm=False):
    """
    Reset a ChromaDB collection by deleting it and recreating it empty
    
    Args:
        collection_name: Name of the ChromaDB collection to reset
        persist_dir: Directory where ChromaDB is persisted
        confirm: If True, proceed without confirmation prompt
        
    Returns:
        bool: True if reset was successful, False otherwise
    """
    try:
        # Create directory for persistence if it doesn't exist
        os.makedirs(persist_dir, exist_ok=True)
        
        # Set up ChromaDB client
        chroma_client = chromadb.PersistentClient(path=persist_dir)
        
        # Check if the collection exists
        collections = chroma_client.list_collections()
        collection_exists = any(c.name == collection_name for c in collections)
        
        if not collection_exists:
            print(f"Collection '{collection_name}' does not exist.")
            return False
        
        # Get document count
        collection = chroma_client.get_collection(collection_name)
        doc_count = len(collection.get()['ids'])
        
        # Confirm with user unless --force flag was used
        if not confirm:
            response = input(f"Are you sure you want to delete all {doc_count} documents in '{collection_name}'? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("Operation cancelled.")
                return False
        
        # Delete the collection
        chroma_client.delete_collection(collection_name)
        print(f"Deleted collection '{collection_name}' with {doc_count} documents.")
        
        # Recreate an empty collection
        chroma_client.create_collection(collection_name)
        print(f"Created empty collection '{collection_name}'.")
        
        return True
    
    except Exception as e:
        print(f"Error resetting collection: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Reset a ChromaDB collection (delete all documents)")
    parser.add_argument("--collection", type=str, default="my_documents", help="ChromaDB collection name to reset")
    parser.add_argument("--db-path", type=str, default="./chroma_db", help="Path to ChromaDB directory")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()
    
    success = reset_collection(
        collection_name=args.collection,
        persist_dir=args.db_path,
        confirm=args.force
    )
    
    if success:
        print("\n✅ Collection reset successfully")
    else:
        print("\n❌ Failed to reset collection")

if __name__ == "__main__":
    main()