#!/usr/bin/env python3
"""
Utility script to load the Mistral 7B dataset into the vector store
"""

from rag import RAGService

def main():
    print("Loading Mistral 7B dataset into vector store...")
    rag_service = RAGService()
    result = rag_service.load_dataset()
    print(f"Result: {result}")

if __name__ == "__main__":
    main() 