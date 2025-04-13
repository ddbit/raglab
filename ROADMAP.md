# RAG System Roadmap

This document outlines the planned features and improvements for the RAG system.

## Current Status

The project is currently in active development with the following core features implemented:

- [x] Basic RAG functionality with ChromaDB vector store
- [x] Support for Ollama LLM integration
- [x] Support for OpenAI API integration
- [x] PDF document ingestion
- [x] Text file ingestion
- [x] Interactive terminal with conversation history
- [x] File injection during conversation
- [x] Command to list indexed documents
- [x] Model information and switching commands
- [x] Test framework for quick validation

## Completed Features

- [x] Basic RAG implementation with ChromaDB
- [x] Ollama integration for local LLMs
- [x] OpenAI integration for cloud LLMs
- [x] PDF document ingestion
- [x] Text document ingestion
- [x] Simple CLI interface
- [x] Interactive terminal with conversation history
- [x] Command to list indexed documents in terminal (/list)
- [x] Command to show current model (/model)
- [x] Command to list models in Ollama (/models)
- [x] Command to switch models (/switch)
- [x] Command to reset conversation context (/reset)
- [x] Basic test framework
- [x] Script to reset vector database

## Future requirements

- [x] Implement a metacommand in the interactive chat to list the documents currently indexed in the vector db
- [x] Implement a metacommand to get the current model
- [x] Implement a metacommand to switch model
- [x] Implement a metacommand to list models in ollama
- [x] Implement a metacommand to reset the context
- [ ] Implement a metacommand to remove a file from the vector db (e.g., /remove 1)

## Technical Debt & Improvements

- [ ] Comprehensive test suite with unit and integration tests
- [ ] Better error handling and reporting
- [ ] Documentation improvements
- [ ] Code refactoring for maintainability
- [ ] Configuration management via config files
- [ ] Proper logging system
- [ ] Containerization with Docker
- [ ] Web UI interface for easier interaction