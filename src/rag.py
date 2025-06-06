import os
from typing import List, Optional, Union
import chromadb
from llama_index.core import Document, Settings, VectorStoreIndex, StorageContext, ServiceContext
from llama_index.core.llms import LLM
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
import hyperparams

# Load environment variables
load_dotenv()

class RAGSystem:
    def __init__(self, 
                 collection_name=hyperparams.DEFAULT_COLLECTION_NAME, 
                 persist_dir=hyperparams.DEFAULT_PERSIST_DIR,
                 model_name=hyperparams.DEFAULT_MODEL_NAME,
                 engine=hyperparams.DEFAULT_ENGINE,
                 api_key=None,       # API key for OpenAI
                 embedding_model_name=hyperparams.EMBEDDING_MODEL,
                 temperature=hyperparams.DEFAULT_TEMPERATURE):
        """Initialize the RAG system with ChromaDB as the vector store and LLM provider
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_dir: Directory to persist ChromaDB
            model_name: Name of the model to use
            engine: LLM engine to use ('ollama' or 'openai')
            api_key: API key for OpenAI (only needed if engine is 'openai')
            embedding_model_name: Name of the embedding model
            temperature: Temperature for LLM generation (0.0 to 1.0)
        """
        # Create directory for persistence if it doesn't exist
        os.makedirs(persist_dir, exist_ok=True)
        
        # Set up ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.chroma_client.get_or_create_collection(collection_name)
        
        # Create vector store from ChromaDB collection
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Configure embeddings
        Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
        
        # Configure LLM based on engine
        if engine.lower() == 'ollama':
            Settings.llm = Ollama(model=model_name, request_timeout=120.0, temperature=temperature)
        elif engine.lower() == 'openai':
            # Use environment variable if api_key is not provided
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key is required when using OpenAI engine. "
                                "Either provide it as api_key parameter or set OPENAI_API_KEY environment variable.")
            Settings.llm = OpenAI(model=model_name, api_key=api_key, temperature=temperature)
        else:
            raise ValueError(f"Unsupported engine: {engine}. Use 'ollama' or 'openai'.")
        
        # Store parameters for reference
        self.model_name = model_name
        self.engine = engine.lower()
        
        # Initialize or load the index
        self.index = self._get_or_create_index()
    
    def _get_or_create_index(self):
        """Get existing index or create new one"""
        # Check if collection has any documents
        if len(self.collection.get()['ids']) > 0:
            return VectorStoreIndex.from_vector_store(
                self.vector_store,
            )
        return VectorStoreIndex([], storage_context=self.storage_context)
    
    def add_documents(self, texts: List[str], metadatas: List[dict] = None):
        """Add documents to the index"""
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        documents = [Document(text=text, metadata=metadata) 
                    for text, metadata in zip(texts, metadatas)]
        
        self.index.insert_nodes(documents)
        return len(documents)
    
    def query(self, query_text: str, similarity_top_k: int = None, retriever_type: str = None):
        """Query the RAG system"""
        # Use provided values or defaults from hyperparams
        similarity_top_k = similarity_top_k if similarity_top_k is not None else hyperparams.TOP_K
        retriever_type = retriever_type if retriever_type is not None else hyperparams.RETRIEVER_TYPE
        
        # Configure retriever based on type
        if retriever_type == "default":
            query_engine = self.index.as_query_engine(similarity_top_k=similarity_top_k)
        elif retriever_type == "sparse":
            from llama_index.core.retrievers import BM25Retriever
            retriever = BM25Retriever.from_defaults(
                index=self.index,
                similarity_top_k=similarity_top_k
            )
            query_engine = self.index.as_query_engine(retriever=retriever)
        elif retriever_type == "dense":
            # Default is already dense retrieval with embeddings
            query_engine = self.index.as_query_engine(similarity_top_k=similarity_top_k)
        elif retriever_type == "hybrid":
            from llama_index.core.retrievers import BM25Retriever
            from llama_index.core.retrievers import VectorIndexRetriever
            from llama_index.core.query_engine import RetrieverQueryEngine
            from llama_index.core.retrievers import HybridRetriever
            
            # Create sparse retriever (BM25)
            sparse_retriever = BM25Retriever.from_defaults(
                index=self.index,
                similarity_top_k=similarity_top_k
            )
            
            # Create dense retriever (default vector retriever)
            dense_retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=similarity_top_k
            )
            
            # Create hybrid retriever
            retriever = HybridRetriever(
                [sparse_retriever, dense_retriever]
            )
            
            query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever
            )
        else:
            # Default to standard retriever
            query_engine = self.index.as_query_engine(similarity_top_k=similarity_top_k)
        
        response = query_engine.query(query_text)
        
        # Return both the response and source documents
        return {
            "answer": str(response),
            "source_nodes": [
                {
                    "text": node.node.text,
                    "score": node.score,
                    "metadata": node.node.metadata
                }
                for node in response.source_nodes
            ]
        }