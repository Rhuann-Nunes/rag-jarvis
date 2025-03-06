#!/usr/bin/env python
import os
import json
import uuid
import requests
from typing import Dict, List, Optional

# API URL
API_URL = os.getenv("API_URL", "http://localhost:8000")

def load_user_data(user_id: str, user_name: str):
    """Load user data with name"""
    payload = {
        "user_id": user_id,
        "user_name": user_name
    }
    
    response = requests.post(f"{API_URL}/api/load-user-data", json=payload)
    return response.json()

def user_query(user_id: str, user_name: str, query_text: str, conversation_history=None, k=3):
    """Query the RAG API with user context"""
    payload = {
        "user_id": user_id,
        "user_name": user_name,
        "query": query_text,
        "conversation_history": conversation_history or [],
        "k": k
    }
    
    response = requests.post(f"{API_URL}/api/user-query", json=payload)
    return response.json()

def user_search(user_id: str, user_name: str, query_text: str, k=3):
    """Search user data"""
    payload = {
        "user_id": user_id,
        "user_name": user_name,
        "query": query_text,
        "k": k
    }
    
    response = requests.post(f"{API_URL}/api/user-search", json=payload)
    return response.json()

def get_user_augmented_prompt(user_id: str, user_name: str, query_text: str, k=3):
    """Get augmented prompt for user"""
    payload = {
        "user_id": user_id,
        "user_name": user_name,
        "query": query_text,
        "k": k
    }
    
    response = requests.post(f"{API_URL}/api/user-augmented-prompt", json=payload)
    return response.json()

def main():
    print("JARVIS - Assistente Pessoal")
    print("--------------------------")
    
    # Get user identification
    user_id = input("\nEntre com seu UUID no Supabase: ")
    user_name = input("Entre com seu nome: ")
    
    # Load user data
    print("\nCarregando dados do usuário...")
    result = load_user_data(user_id, user_name)
    print(f"Resultado: {result}")
    
    # Start JARVIS conversation
    conversation_history = []
    
    print("\n=== Exemplos de consultas para testar tarefas recorrentes ===")
    print("- Quais são minhas tarefas para amanhã?")
    print("- O que tenho planejado para esta semana?")
    print("- Mostre minhas tarefas recorrentes pendentes")
    print("- Tenho alguma tarefa que se repete mensalmente?")
    print("=================================================\n")
    
    while True:
        # Get query from user
        query_text = input("\nDigite sua pergunta (ou 'sair' para encerrar): ")
        
        if query_text.lower() == "sair":
            break
            
        # Send query to API
        print("Consultando o JARVIS...")
        result = user_query(user_id, user_name, query_text, conversation_history)
        
        # Print answer
        print("\nJARVIS:")
        print(result["answer"])
        
        # Update conversation history
        conversation_history.append({"role": "user", "content": query_text})
        conversation_history.append({"role": "assistant", "content": result["answer"]})
        
        # Option to see the prompts
        answer = input("\nDeseja ver os prompts? (s/n): ")
        if answer.lower() == "s":
            print("\nPrompt do Sistema:")
            print(result.get("system_prompt", "Não disponível"))
            print("\nPrompt Aumentado:")
            print(result.get("augmented_prompt", "Não disponível"))

if __name__ == "__main__":
    main() 