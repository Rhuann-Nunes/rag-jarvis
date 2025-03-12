#!/usr/bin/env python
import os
import json
import requests
from typing import Dict, List, Optional
import time

# API URL
API_URL = os.getenv("API_URL", "http://localhost:8000")

def sales_query(query_text: str, conversation_history=None, k=3):
    """Query the Sales RAG API with persuasive content"""
    payload = {
        "query": query_text,
        "conversation_history": conversation_history or [],
        "k": k
    }
    
    response = requests.post(f"{API_URL}/api/sales-query", json=payload)
    return response.json()

def sales_search(query_text: str, k=3):
    """Search sales content"""
    payload = {
        "query": query_text,
        "k": k
    }
    
    response = requests.post(f"{API_URL}/api/sales-search", json=payload)
    return response.json()

def reload_sales_content():
    """Reload the sales content from markdown file"""
    response = requests.post(f"{API_URL}/api/reload-sales-content")
    return response.json()

def sample_objections():
    """Lista de objeções comuns de vendas para testar o RAG"""
    return [
        "Quanto custa o JARVIS? Parece caro demais.",
        "Por que eu usaria o JARVIS ao invés do meu sistema atual?",
        "Como o JARVIS é melhor que o Todoist ou Microsoft To-Do?",
        "Não tenho certeza se vale a pena investir nesse aplicativo. Pode me convencer?",
        "O que acontece se eu não quiser usar o WhatsApp para receber notificações?",
        "Como sei que é seguro dar meus dados ao JARVIS?",
        "Funciona offline? Preciso de algo que funcione sem internet.",
        "Parece complicado demais. Não tenho tempo para aprender outra ferramenta.",
        "Que garantias eu tenho que o JARVIS realmente vai melhorar minha produtividade?",
        "O que acontece quando meu período gratuito acabar? Vou perder meus dados?",
    ]

def auto_demo():
    """Executa uma demonstração automática com objeções comuns"""
    print("\n===== TESTE AUTOMÁTICO DO VENDEDOR JARVIS =====\n")
    
    # Recarregar conteúdo de vendas
    print("Recarregando conteúdo de vendas...")
    reload_result = reload_sales_content()
    print(f"Resultado: {reload_result}\n")
    
    # Testar objeções
    objections = sample_objections()
    conversation_history = []
    
    for i, objection in enumerate(objections, 1):
        print(f"\n--- OBJEÇÃO {i}/{len(objections)} ---")
        print(f"Cliente: {objection}")
        
        # Enviar para a API
        start_time = time.time()
        result = sales_query(objection, conversation_history)
        end_time = time.time()
        
        # Mostrar resposta
        print(f"\nJARVIS Sales ({(end_time - start_time):.2f}s):")
        print(result["answer"])
        print("\n" + "-"*50)
        
        # Atualizar histórico
        conversation_history.append({"role": "user", "content": objection})
        conversation_history.append({"role": "assistant", "content": result["answer"]})
        
        # Pausa entre consultas para não sobrecarregar a API
        if i < len(objections):
            time.sleep(1)
    
    print("\n===== FIM DA DEMONSTRAÇÃO =====\n")

def main():
    """Interactive client for the Sales RAG API"""
    print("JARVIS Sales - Cliente de Demonstração")
    print("--------------------------------------")
    
    # Inicialização
    print("\nOpções disponíveis:")
    print("1. Conversar com o vendedor JARVIS")
    print("2. Pesquisar conteúdo de vendas")
    print("3. Recarregar conteúdo de vendas")
    print("4. Executar demonstração automática")
    print("5. Sair")
    
    while True:
        choice = input("\nEscolha uma opção (1-5): ")
        
        if choice == "1":
            # Conversar com JARVIS Sales
            conversation_history = []
            
            while True:
                query_text = input("\nDigite sua pergunta (ou 'voltar' para retornar): ")
                
                if query_text.lower() == "voltar":
                    break
                    
                print("Consultando o JARVIS Sales...")
                result = sales_query(query_text, conversation_history)
                
                print("\nJARVIS Sales:")
                print(result["answer"])
                
                # Atualizar histórico
                conversation_history.append({"role": "user", "content": query_text})
                conversation_history.append({"role": "assistant", "content": result["answer"]})
                
                # Opção para ver prompts
                show_prompt = input("\nDeseja ver o prompt? (s/n): ")
                if show_prompt.lower() == "s":
                    print("\nPrompt Aumentado:")
                    print(result.get("augmented_prompt", "Não disponível"))
        
        elif choice == "2":
            # Pesquisar conteúdo
            query_text = input("\nDigite o termo de pesquisa: ")
            k = input("Quantos resultados deseja ver? (padrão: 3): ")
            k = int(k) if k.isdigit() else 3
            
            print("Pesquisando conteúdo...")
            results = sales_search(query_text, k)
            
            if "results" in results:
                print(f"\nEncontrados {len(results['results'])} resultados:")
                for i, result in enumerate(results["results"], 1):
                    print(f"\n--- Resultado {i} (Score: {result['score']:.2f}) ---")
                    print(f"Seção: {result['metadata'].get('section', 'N/A')}")
                    print(result["text"])
            else:
                print("\nErro na pesquisa:", results.get("message", "Erro desconhecido"))
        
        elif choice == "3":
            # Recarregar conteúdo
            print("Recarregando conteúdo de vendas...")
            result = reload_sales_content()
            print(f"Resultado: {result}")
        
        elif choice == "4":
            # Executar demo automática
            auto_demo()
        
        elif choice == "5":
            # Sair
            print("\nEncerrando cliente. Até logo!")
            break
        
        else:
            print("Opção inválida. Por favor, escolha uma opção de 1 a 5.")

if __name__ == "__main__":
    main() 