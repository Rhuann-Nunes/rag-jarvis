import os
from typing import List, Dict, Any, Optional
import logging

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from groq import Groq

import config
from user_data_service import UserDataService

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGService:
    """Retrieval-Augmented Generation service using Groq"""

    def __init__(self):
        # Initialize embedding model
        self.embedding_model = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
        )
        
        # Initialize Groq client
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Initialize user data service
        self.user_data_service = UserDataService(use_in_memory=True, embedding_model=self.embedding_model)
        
        logger.info(f"RAG Service initialized with model: {config.GROQ_MODEL_NAME}")
    
    def load_user_data(self, user_id: str, user_name: str = ""):
        """Load user data from Supabase and store in vector database"""
        user_data = self.user_data_service.load_user_data(user_id, user_name)
        return {"status": "success", "message": "User data loaded successfully", "user_data": user_data}
    
    def search_user_data(self, user_id: str, query: str, k: int = 3):
        """Search for relevant user data based on the query"""
        results = self.user_data_service.search_user_data(user_id, query, limit=k)
        return results
    
    def generate_user_augmented_prompt(self, user_id: str, user_name: str, form_of_address: str, query: str, k: int = 3):
        """Generate an augmented prompt with user context"""
        # Get relevant user documents
        user_results = self.search_user_data(user_id, query, k=k)
        
        # Extract text from results
        user_context = "\n".join([result["text"] for result in user_results])
        
        # Create personalized system prompt
        system_prompt = f"""Você é JARVIS, um assistente pessoal inteligente que ajuda {form_of_address or ''} {user_name or 'o usuário'} 
com suas tarefas e projetos. Você tem acesso aos dados pessoais do usuário, incluindo projetos e tarefas.
Seja atencioso, prestativo e sempre trate o usuário por {form_of_address or ''} {user_name or 'usuário'}.
Utilize os dados do usuário para personalizar suas respostas e fazer recomendações úteis.

IMPORTANTE - CONTEXTO TEMPORAL:
Você tem acesso à data e hora atual no fuso horário de Brasília (UTC-3). Use essa informação para:
1. Interpretar corretamente referências temporais como "hoje", "amanhã", "próxima semana"
2. Calcular prazos e datas relativas ao momento atual
3. Identificar tarefas atrasadas ou próximas do vencimento
4. Fornecer respostas precisas sobre o tempo restante até os prazos

IMPORTANTE - TAREFAS RECORRENTES:
Algumas tarefas possuem recorrência, identificadas pelas propriedades "recurrence_type" e "recurrence_interval".
Quando o usuário perguntar sobre tarefas em datas específicas ou em intervalos de tempo, você deve:
1. Identificar tarefas recorrentes não concluídas
2. Calcular as ocorrências dessas tarefas baseadas no tipo de recorrência:
   - "daily": a tarefa repete a cada X dias (onde X é o recurrence_interval)
   - "weekly": a tarefa repete a cada X semanas
   - "monthly": a tarefa repete a cada X meses
   - "yearly": a tarefa repete a cada X anos
3. Apenas considere as recorrências para tarefas não concluídas (completed = false)
4. Inclua as ocorrências calculadas que caem dentro do período solicitado pelo usuário

Se o usuário perguntar por tarefas em uma data específica (ex: "quais são minhas tarefas para amanhã?") ou em um intervalo (ex: "tarefas desta semana"), leve em consideração as tarefas recorrentes não concluídas que teriam ocorrências nesse período.

Todas as datas e horários fornecidos já estão no fuso horário de Brasília (UTC-3).
"""
        
        # Create augmented prompt with user context
        augmented_prompt = f"""Use o contexto abaixo para responder à pergunta de {form_of_address or ''} {user_name or 'do usuário'}.

Contexto sobre os dados do usuário:
{user_context}

Pergunta: {query}"""
        
        return system_prompt, augmented_prompt
    
    def _convert_messages_to_groq_format(self, messages):
        """Convert LangChain message format to Groq API format"""
        groq_messages = []
        for message in messages:
            if isinstance(message, SystemMessage):
                groq_messages.append({"role": "system", "content": message.content})
            elif isinstance(message, HumanMessage):
                groq_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                groq_messages.append({"role": "assistant", "content": message.content})
        return groq_messages

    def answer_user_query(self, user_id: str, user_name: str, form_of_address: str, query: str, 
                        conversation_history: List[Dict[str, str]] = None, k: int = 3):
        """Answer a query using the RAG approach with user context"""
        try:
            # Generate personalized system prompt and augmented prompt
            system_prompt, augmented_prompt = self.generate_user_augmented_prompt(
                user_id, user_name, form_of_address, query, k
            )
            
            # Initialize messages with personalized system prompt
            messages = [SystemMessage(content=system_prompt)]
            
            # Add conversation history if provided
            if conversation_history:
                for entry in conversation_history:
                    if entry["role"] == "user":
                        messages.append(HumanMessage(content=entry["content"]))
                    elif entry["role"] == "assistant":
                        messages.append(AIMessage(content=entry["content"]))
            
            # Add current query with context
            messages.append(HumanMessage(content=augmented_prompt))
            
            # Convert to Groq format
            groq_messages = self._convert_messages_to_groq_format(messages)
            
            # Invoke the model directly using Groq client
            logger.info(f"Invocando modelo Groq para usuário {user_id} com {len(groq_messages)} mensagens")
            completion = self.groq_client.chat.completions.create(
                model=config.GROQ_MODEL_NAME,
                messages=groq_messages,
                temperature=0.6,
                max_completion_tokens=1024,
                top_p=0.95,
                stream=False,
                reasoning_format="hidden"
            )
            response_content = completion.choices[0].message.content
            logger.info("Resposta do modelo recebida com sucesso")
            
            return {
                "answer": response_content,
                "augmented_prompt": augmented_prompt,
                "system_prompt": system_prompt
            }
        except Exception as e:
            logger.error(f"Erro ao responder consulta do usuário {user_id}: {e}")
            return {
                "answer": f"Desculpe, ocorreu um erro ao processar sua consulta: {str(e)}",
                "augmented_prompt": "",
                "system_prompt": ""
            }
    
    def _convert_history_to_messages(self, conversation_history):
        """Convert conversation history to LangChain message format"""
        messages = [
            SystemMessage(content="Você é um assistente útil que responde perguntas com base no contexto fornecido.")
        ]
        
        for entry in conversation_history:
            if entry["role"] == "user":
                messages.append(HumanMessage(content=entry["content"]))
            elif entry["role"] == "assistant":
                messages.append(AIMessage(content=entry["content"]))
        
        return messages 