import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from rag import RAGSystem
import hyperparams

DEFAULT_CONTEXT_FILE = os.path.expanduser("~/.context")

class RAGTerminal:
    def __init__(
        self,
        collection_name: str = hyperparams.DEFAULT_COLLECTION_NAME,
        model_name: str = hyperparams.DEFAULT_MODEL_NAME,
        engine: str = hyperparams.DEFAULT_ENGINE,
        api_key: Optional[str] = None,
        context_file: str = DEFAULT_CONTEXT_FILE,
        max_history: int = 10,
        temperature: float = hyperparams.DEFAULT_TEMPERATURE,
        top_k: int = hyperparams.TOP_K,
        retriever_type: str = hyperparams.RETRIEVER_TYPE
    ):
        """Initialize the RAG terminal app"""
        
        self.rag = RAGSystem(
            collection_name=collection_name,
            model_name=model_name,
            engine=engine,
            api_key=api_key,
            temperature=temperature
        )
        
        self.top_k = top_k
        self.retriever_type = retriever_type
        
        self.context_file = context_file
        self.max_history = max_history
        self.conversation_history = []
        
        # Load conversation history if it exists
        self._load_context()
        
    def _load_context(self) -> None:
        """Load conversation history from context file"""
        try:
            if os.path.exists(self.context_file):
                with open(self.context_file, 'r', encoding='utf-8') as f:
                    self.conversation_history = json.load(f)
                    
                # Trim history if exceeds max_history
                if len(self.conversation_history) > self.max_history:
                    self.conversation_history = self.conversation_history[-self.max_history:]
                
                print(f"Loaded {len(self.conversation_history)} previous messages from context")
        except Exception as e:
            print(f"Error loading context: {str(e)}")
            self.conversation_history = []
    
    def _save_context(self) -> None:
        """Save conversation history to context file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.context_file)), exist_ok=True)
            
            with open(self.context_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving context: {str(e)}")
    
    def _format_history_for_prompt(self) -> str:
        """Format conversation history for inclusion in prompt"""
        formatted = ""
        
        for entry in self.conversation_history:
            role = entry.get("role", "")
            content = entry.get("content", "")
            
            if role.lower() == "user":
                formatted += f"User: {content}\n\n"
            elif role.lower() == "assistant":
                formatted += f"Assistant: {content}\n\n"
        
        return formatted.strip()
    
    def handle_command(self, command: str) -> bool:
        """Handle special commands, return True if should exit loop"""
        
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        
        if cmd == "/bye":
            print("Goodbye!")
            return True
            
        elif cmd == "/clear":
            self.conversation_history = []
            self._save_context()
            print("Conversation history cleared.")
        
        elif cmd == "/reset":
            self._reset_context()
        
        elif cmd == "/list":
            self._list_documents()
            
        elif cmd == "/model":
            self._show_current_model()
            
        elif cmd == "/models":
            self._list_models()
            
        elif cmd == "/switch" and len(parts) > 1:
            model_name = parts[1].strip()
            self._switch_model(model_name)
            
        elif cmd == "/inject" and len(parts) > 1:
            path = parts[1].strip()
            
            # Expand user directory if needed
            path = os.path.expanduser(path)
            
            # Check if path exists
            if not os.path.exists(path):
                print(f"Error: File '{path}' does not exist.")
                return False
                
            try:
                # Use the process_single_file function from build.py
                from build import process_single_file
                
                # Process the file directly
                chunk_size = args.chunk_size if hasattr(args, 'chunk_size') else hyperparams.CHUNK_SIZE
                chunk_overlap = args.chunk_overlap if hasattr(args, 'chunk_overlap') else hyperparams.CHUNK_OVERLAP
                
                total_chunks = process_single_file(
                    file_path=path,
                    rag_system=self.rag,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                print(f"Successfully injected file '{os.path.basename(path)}' with {total_chunks} chunks.")
                
            except Exception as e:
                print(f"Error injecting file: {str(e)}")
                
                
        elif cmd == "/help":
            print("\nAvailable commands:")
            print("  /bye          - Exit the program")
            print("  /clear        - Clear conversation history")
            print("  /reset        - Reset conversation context without clearing history")
            print("  /list         - List all documents in the vector database")
            print("  /model        - Show current model information")
            print("  /models       - List available models in Ollama (if using Ollama engine)")
            print("  /switch <model> - Switch to a different model")
            print("  /inject <path> - Inject a file into the vector database")
            print("  /help         - Show this help message")
            
        else:
            print(f"Unknown command: {cmd}. Type /help for available commands.")
            
        return False
        
    def _reset_context(self):
        """Reset the conversation context without clearing history"""
        try:
            # Save the conversation history
            old_history = self.conversation_history.copy()
            
            # Print a divider in the history
            self.conversation_history.append({"role": "system", "content": "--- Context Reset ---"})
            self._save_context()
            
            print("Conversation context has been reset. New questions will not reference previous context.")
            print("History is preserved for your reference, but the LLM will not use it for context.")
        except Exception as e:
            print(f"Error resetting context: {str(e)}")
            
    def _list_models(self):
        """List available models in Ollama"""
        try:
            if self.rag.engine.lower() != 'ollama':
                print(f"\nThis command is only available when using the Ollama engine.")
                print(f"Current engine: {self.rag.engine}")
                return
                
            # Run ollama list command using subprocess
            import subprocess
            import json
            
            try:
                # Get Ollama host from environment or use default
                ollama_host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
                
                # Use curl to get models from Ollama API
                cmd = ['curl', '-s', f'{ollama_host}/api/tags']
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"\nError fetching models from Ollama: {result.stderr}")
                    return
                    
                # Parse JSON response
                try:
                    models_data = json.loads(result.stdout)
                    models = models_data.get('models', [])
                    
                    if not models:
                        print("\nNo models found in Ollama.")
                        return
                        
                    print("\n=== Available Models in Ollama ===")
                    for i, model in enumerate(models):
                        name = model.get('name', 'Unknown')
                        size = model.get('size', 0) / (1024 * 1024 * 1024)  # Convert to GB
                        modified = model.get('modified', 'Unknown')
                        
                        # Mark current model
                        current = " (current)" if name == self.rag.model_name else ""
                        
                        print(f"[{i+1}] {name}{current}")
                        print(f"    Size: {size:.2f} GB")
                        print(f"    Modified: {modified}")
                    
                    print("\nTo switch models, use: /switch <model_name>")
                except json.JSONDecodeError:
                    print(f"\nError parsing Ollama response: Invalid JSON")
                    return
                
            except FileNotFoundError:
                print("\nError: 'curl' command not found. Please install curl to use this feature.")
                return
                
        except Exception as e:
            print(f"Error listing models: {str(e)}")
            
    def _switch_model(self, model_name):
        """Switch to a different model"""
        try:
            if not model_name:
                print("Please specify a model name: /switch <model_name>")
                return
                
            print(f"\nSwitching model from {self.rag.model_name} to {model_name}...")
            
            # OpenAI models can be switched directly
            if self.rag.engine.lower() == 'openai':
                old_model = self.rag.model_name
                
                # Update the model in the RAG system
                from llama_index.llms.openai import OpenAI
                self.rag.model_name = model_name
                self.rag.Settings.llm = OpenAI(model=model_name, api_key=self.rag.api_key)
                
                print(f"Successfully switched from {old_model} to {model_name} (OpenAI)")
                return
                
            # For Ollama, verify the model exists first
            if self.rag.engine.lower() == 'ollama':
                import subprocess
                import json
                
                # Get Ollama host from environment or use default
                ollama_host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
                
                # Check if the model exists
                cmd = ['curl', '-s', f'{ollama_host}/api/tags']
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"Error checking models in Ollama: {result.stderr}")
                    return
                
                try:
                    models_data = json.loads(result.stdout)
                    models = models_data.get('models', [])
                    model_names = [m.get('name') for m in models]
                    
                    if model_name not in model_names:
                        print(f"Model '{model_name}' not found in Ollama.")
                        print("Available models:")
                        for m in model_names:
                            print(f"  - {m}")
                        return
                    
                    # Update the model in the RAG system
                    old_model = self.rag.model_name
                    
                    # Update the model 
                    from llama_index.llms.ollama import Ollama
                    self.rag.model_name = model_name
                    self.rag.Settings.llm = Ollama(model=model_name, request_timeout=120.0)
                    
                    print(f"Successfully switched from {old_model} to {model_name} (Ollama)")
                    
                except json.JSONDecodeError:
                    print("Error parsing Ollama response: Invalid JSON")
                    return
            
        except Exception as e:
            print(f"Error switching model: {str(e)}")
        
    def _show_current_model(self):
        """Show information about the current model"""
        try:
            print("\n=== Current Model Information ===")
            print(f"Model name: {self.rag.model_name}")
            print(f"Engine: {self.rag.engine}")
            
            # Additional info depending on engine
            if self.rag.engine.lower() == 'ollama':
                print(f"Ollama endpoint: {os.environ.get('OLLAMA_HOST', 'http://localhost:11434')}")
            elif self.rag.engine.lower() == 'openai':
                print("Using OpenAI API")
                # Don't print the API key for security reasons
                has_key = hasattr(self.rag, 'api_key') and self.rag.api_key is not None
                print(f"API key: {'Configured' if has_key else 'Not configured (using environment variable)'}")
            
            print("\nTo switch models, restart the application with different parameters:")
            print("  --model <model_name> --engine <engine_name>")
        except Exception as e:
            print(f"Error retrieving model information: {str(e)}")
        
    def _list_documents(self):
        """List all documents in the vector database"""
        try:
            # Get all documents from the collection
            collection_data = self.rag.collection.get()
            
            if not collection_data or not collection_data.get('metadatas') or len(collection_data['metadatas']) == 0:
                print("\nNo documents found in the vector database.")
                return
            
            # Extract unique document sources from metadata
            documents = {}
            for i, metadata in enumerate(collection_data['metadatas']):
                if not metadata:
                    continue
                    
                source = metadata.get('source', 'Unknown source')
                filename = metadata.get('filename', os.path.basename(source) if source != 'Unknown source' else 'Unknown')
                filetype = metadata.get('filetype', 'unknown')
                page = metadata.get('page')
                
                # Use filename as key
                if filename not in documents:
                    documents[filename] = {
                        'source': source,
                        'filetype': filetype,
                        'chunks': 0,
                        'pages': set()
                    }
                
                documents[filename]['chunks'] += 1
                if page:
                    documents[filename]['pages'].add(page)
            
            # Print the results
            print("\n=== Documents in Vector Database ===")
            print(f"Collection: {self.rag.collection.name}")
            print(f"Total chunks: {len(collection_data['ids'])}")
            print("\nDocuments:")
            
            for i, (filename, info) in enumerate(documents.items()):
                pages_str = ""
                if info['pages']:
                    pages_list = sorted(list(info['pages']))
                    if len(pages_list) <= 10:
                        pages_str = f" - Pages: {', '.join(str(p) for p in pages_list)}"
                    else:
                        pages_str = f" - {len(pages_list)} pages"
                
                print(f"[{i+1}] {filename} ({info['filetype']}) - {info['chunks']} chunks{pages_str}")
                print(f"    Source: {info['source']}")
            
            print("\nUse these documents in your queries.")
        
        except Exception as e:
            print(f"Error listing documents: {str(e)}")
    
    def process_query(self, user_input: str) -> Dict:
        """Process a user query using RAG and conversation history"""
        
        # Add conversation history context if available
        history_context = self._format_history_for_prompt()
        
        # Combine history with current query
        if history_context:
            enhanced_query = f"{history_context}\n\nUser: {user_input}\n\nAssistant:"
        else:
            enhanced_query = user_input
        
        # Get response from RAG system
        result = self.rag.query(
            enhanced_query, 
            similarity_top_k=self.top_k,
            retriever_type=self.retriever_type
        )
        
        # Store the exchange in history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": result["answer"]})
        
        # Trim history if needed
        if len(self.conversation_history) > self.max_history * 2:  # *2 because each interaction is 2 entries
            self.conversation_history = self.conversation_history[-self.max_history * 2:]
        
        # Save updated context
        self._save_context()
        
        return result
    
    def run(self) -> None:
        """Run the interactive terminal loop"""
        
        print("\n===== RAG Terminal =====")
        print("Type your questions or use special commands:")
        print("  /list         - List all documents in the vector database")
        print("  /model        - Show current model information")
        print("  /models       - List available models in Ollama")
        print("  /switch <model> - Switch to a different model")
        print("  /inject <path> - Inject a file into the vector database")
        print("  /clear        - Clear conversation history")
        print("  /reset        - Reset conversation context")
        print("  /bye          - Exit the program")
        print("  /help         - Show all commands")
        print("========================\n")
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                # Skip empty inputs
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.startswith('/'):
                    if self.handle_command(user_input):
                        break
                    continue
                
                # Process regular queries
                result = self.process_query(user_input)
                
                # Display answer
                print("\nAnswer:", result["answer"])
                
                # Display sources if available
                if result.get("source_nodes") and len(result["source_nodes"]) > 0:
                    print("\nSources:")
                    for i, source in enumerate(result["source_nodes"]):
                        source_info = f"{source.get('metadata', {}).get('filename', 'unknown')}"
                        if source.get('metadata', {}).get('page'):
                            source_info += f" (page {source['metadata']['page']})"
                        print(f"\n[{i+1}] {source_info} (Score: {source['score']:.4f})")
                        # Print a short snippet of the text
                        text = source.get('text', '')
                        if text:
                            print(f"    {text[:100]}..." if len(text) > 100 else f"    {text}")
                
            except KeyboardInterrupt:
                print("\nInterrupted. Type /bye to exit.")
            except Exception as e:
                print(f"\nError: {str(e)}")

def query_once():
    """Original query function for backward compatibility"""
    parser = argparse.ArgumentParser(description="Query the RAG system")
    parser.add_argument("--query", type=str, required=True, help="Question to ask")
    parser.add_argument("--model", type=str, default=hyperparams.DEFAULT_MODEL_NAME, help="Model name to use")
    parser.add_argument("--engine", type=str, default=hyperparams.DEFAULT_ENGINE, choices=["ollama", "openai"],
                       help="LLM engine to use (ollama or openai)")
    parser.add_argument("--api-key", type=str, help="API key for OpenAI (if using OpenAI engine)")
    parser.add_argument("--collection", type=str, default=hyperparams.DEFAULT_COLLECTION_NAME, help="ChromaDB collection name")
    parser.add_argument("--top-k", type=int, default=hyperparams.TOP_K, help="Number of documents to retrieve")
    parser.add_argument("--temperature", type=float, default=hyperparams.DEFAULT_TEMPERATURE, help="Temperature for generation")
    parser.add_argument("--retriever", type=str, default=hyperparams.RETRIEVER_TYPE, 
                       choices=["default", "sparse", "dense", "hybrid"], help="Type of retriever to use")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    args = parser.parse_args()
    
    # Initialize RAG system
    rag = RAGSystem(
        collection_name=args.collection,
        model_name=args.model,
        engine=args.engine,
        api_key=args.api_key,
        temperature=args.temperature
    )
    
    # Query the system
    result = rag.query(
        args.query, 
        similarity_top_k=args.top_k,
        retriever_type=args.retriever
    )
    
    # Output the results
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("\nQuery:", args.query)
        print("\nAnswer:", result["answer"])
        print("\nSources:")
        for i, source in enumerate(result["source_nodes"]):
            print(f"\n[{i+1}] {source['text'][:200]}..." if len(source['text']) > 200 else f"\n[{i+1}] {source['text']}")
            print(f"    Score: {source['score']:.4f}")
            if "source" in source["metadata"]:
                print(f"    Source: {source['metadata']['source']}")

def main():
    parser = argparse.ArgumentParser(description="RAG Terminal or Query Tool")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive terminal mode")
    parser.add_argument("--query", "-q", type=str, help="Question to ask (non-interactive mode)")
    parser.add_argument("--model", type=str, default=hyperparams.DEFAULT_MODEL_NAME, help="Model name to use")
    parser.add_argument("--engine", type=str, default=hyperparams.DEFAULT_ENGINE, choices=["ollama", "openai"],
                       help="LLM engine to use (ollama or openai)")
    parser.add_argument("--api-key", type=str, help="API key for OpenAI (if using OpenAI engine)")
    parser.add_argument("--collection", type=str, default=hyperparams.DEFAULT_COLLECTION_NAME, help="ChromaDB collection name")
    parser.add_argument("--context-file", type=str, default=DEFAULT_CONTEXT_FILE, 
                       help="File to store conversation context")
    parser.add_argument("--max-history", type=int, default=5, 
                       help="Maximum number of message pairs to keep in history")
    parser.add_argument("--top-k", type=int, default=hyperparams.TOP_K, 
                       help="Number of documents to retrieve")
    parser.add_argument("--temperature", type=float, default=hyperparams.DEFAULT_TEMPERATURE, 
                       help="Temperature for generation (0.0 to 1.0)")
    parser.add_argument("--retriever", type=str, default=hyperparams.RETRIEVER_TYPE, 
                       choices=["default", "sparse", "dense", "hybrid"], help="Type of retriever to use")
    parser.add_argument("--knowledge-dir", type=str, default=hyperparams.KNOWLEDGE_BASE_DIR,
                       help="Directory containing knowledge base files")
    parser.add_argument("--chunk-size", type=int, default=hyperparams.CHUNK_SIZE, 
                       help="Size of document chunks (for injection)")
    parser.add_argument("--chunk-overlap", type=int, default=hyperparams.CHUNK_OVERLAP, 
                       help="Overlap between chunks (for injection)")
    parser.add_argument("--json", action="store_true", 
                       help="Output in JSON format (non-interactive mode)")
    
    args, unknown = parser.parse_known_args()
    
    # If explicit --query is provided or --interactive is not set, and no unknown args
    if args.query or (not args.interactive and not unknown):
        # For backward compatibility, if --query is provided, run the old query function
        if args.query:
            # Reconstruct sys.argv to include only the relevant arguments for query_once
            import sys
            sys.argv = [sys.argv[0]] + [f"--query={args.query}"]
            if args.model != "phi3":
                sys.argv.append(f"--model={args.model}")
            if args.engine != "ollama":
                sys.argv.append(f"--engine={args.engine}")
            if args.api_key:
                sys.argv.append(f"--api-key={args.api_key}")
            if args.collection != "my_documents":
                sys.argv.append(f"--collection={args.collection}")
            if args.top_k != hyperparams.TOP_K:
                sys.argv.append(f"--top-k={args.top_k}")
            if args.temperature != hyperparams.DEFAULT_TEMPERATURE:
                sys.argv.append(f"--temperature={args.temperature}")
            if args.retriever != hyperparams.RETRIEVER_TYPE:
                sys.argv.append(f"--retriever={args.retriever}")
            if args.json:
                sys.argv.append("--json")
            
            query_once()
        else:
            # If no query and no interactive flag, print help
            parser.print_help()
    else:
        # Run in interactive mode
        terminal = RAGTerminal(
            collection_name=args.collection,
            model_name=args.model,
            engine=args.engine,
            api_key=args.api_key,
            context_file=args.context_file,
            max_history=args.max_history,
            temperature=args.temperature,
            top_k=args.top_k,
            retriever_type=args.retriever
        )
        
        terminal.run()

if __name__ == "__main__":
    main()