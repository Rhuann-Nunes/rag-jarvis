from fastapi import FastAPI, Depends, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import uuid
import os
import logging
import traceback

# Import RAG services
from rag import RAGService
from sales_rag import JARVISAssistantService
from supabase_client import SupabaseClient

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create app
app = FastAPI(
    title="RAG Jarvis API",
    description="Retrieval-Augmented Generation API usando Groq",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG services and Supabase client
rag_service = None
jarvis_assistant = None
supabase_client = None

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    global rag_service, jarvis_assistant, supabase_client
    rag_service = RAGService()
    jarvis_assistant = JARVISAssistantService()
    supabase_client = SupabaseClient()

# Define models
class UserQueryRequest(BaseModel):
    user_id: str = Field(..., description="The Supabase UUID of the user")
    user_name: str = Field(..., description="The name of the user")
    query: str
    conversation_history: Optional[List[Dict[str, str]]] = None
    k: Optional[int] = 3  # Number of documents to retrieve

class JarvisQueryRequest(BaseModel):
    query: str
    conversation_history: Optional[List[Dict[str, str]]] = None

class QueryResponse(BaseModel):
    answer: str
    augmented_prompt: Optional[str] = None
    system_prompt: Optional[str] = None

class LoadUserDataRequest(BaseModel):
    user_id: str = Field(..., description="The Supabase UUID of the user")
    user_name: str = Field(..., description="The name of the user")

# API endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    """Return HTML interface for testing"""
    return FileResponse("static/index.html")

@app.get("/api")
async def api_root():
    """API root endpoint"""
    return {
        "message": "RAG Jarvis API is running",
        "docs_url": "/docs",
        "version": "0.1.0"
    }

# JARVIS USER RAG ENDPOINTS

@app.post("/api/load-user-data")
async def load_user_data(request: LoadUserDataRequest):
    """Load user data from Supabase"""
    global rag_service
    if not rag_service:
        logger.error("RAG service not initialized")
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    
    try:
        # Validate UUID format
        try:
            uuid_obj = uuid.UUID(request.user_id)
        except ValueError:
            logger.error(f"UUID inválido: {request.user_id}")
            raise HTTPException(status_code=400, detail="Invalid user_id format. Must be a valid UUID.")
        
        # Load user data with provided user_name
        result = rag_service.load_user_data(request.user_id, request.user_name)
        return result
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Erro ao carregar dados do usuário: {str(e)}\n{error_trace}")
        return {"status": "error", "message": str(e)}

@app.post("/api/user-query", response_model=QueryResponse)
async def user_query(request: UserQueryRequest):
    """Answer a query using personalized RAG approach for a specific user"""
    global rag_service, supabase_client
    if not rag_service:
        logger.error("RAG service not initialized")
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    if not supabase_client:
        logger.error("Supabase client not initialized")
        raise HTTPException(status_code=500, detail="Supabase client not initialized")
    
    try:
        # Validate UUID format
        try:
            uuid_obj = uuid.UUID(request.user_id)
            logger.info(f"Processando consulta de usuário: {request.user_id}")
        except ValueError:
            logger.error(f"UUID inválido: {request.user_id}")
            raise HTTPException(status_code=400, detail="Invalid user_id format. Must be a valid UUID.")
        
        # Get user data using the provided user_name
        user_data = supabase_client.get_all_user_data(request.user_id, request.user_name)
        
        user_name = user_data.get("user_name", "")
        form_of_address = user_data.get("form_of_address", "")
        logger.info(f"Dados do usuário obtidos: nome={user_name}, tratamento={form_of_address}")
        
        # Answer query with personalized context
        result = rag_service.answer_user_query(
            user_id=request.user_id,
            user_name=user_name,
            form_of_address=form_of_address,
            query=request.query,
            conversation_history=request.conversation_history,
            k=request.k
        )
        logger.info("Consulta do usuário processada com sucesso")
        
        return QueryResponse(
            answer=result["answer"],
            augmented_prompt=result["augmented_prompt"],
            system_prompt=result["system_prompt"]
        )
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Erro ao processar consulta do usuário: {str(e)}\n{error_trace}")
        return QueryResponse(
            answer=f"Desculpe, ocorreu um erro ao processar sua consulta: {str(e)}",
            augmented_prompt="",
            system_prompt=""
        )

@app.post("/api/user-search")
async def user_search(
    user_id: str = Body(..., embed=True),
    user_name: str = Body(..., embed=True),
    query: str = Body(..., embed=True),
    k: int = Body(3, embed=True)
):
    """Search for relevant user data based on the query"""
    global rag_service
    if not rag_service:
        logger.error("RAG service not initialized")
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    
    try:
        # Validate UUID format
        try:
            uuid_obj = uuid.UUID(user_id)
        except ValueError:
            logger.error(f"UUID inválido: {user_id}")
            raise HTTPException(status_code=400, detail="Invalid user_id format. Must be a valid UUID.")
        
        # First, load user data
        rag_service.load_user_data(user_id, user_name)
        
        # Then search
        results = rag_service.search_user_data(user_id, query, k=k)
        return {"results": results}
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Erro ao buscar dados do usuário: {str(e)}\n{error_trace}")
        return {"status": "error", "message": str(e)}

@app.post("/api/user-augmented-prompt")
async def get_user_augmented_prompt(
    user_id: str = Body(..., embed=True),
    user_name: str = Body(..., embed=True),
    query: str = Body(..., embed=True),
    k: int = Body(3, embed=True)
):
    """Generate an augmented prompt with user context"""
    global rag_service
    if not rag_service:
        logger.error("RAG service not initialized")
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    
    try:
        # Validate UUID format
        try:
            uuid_obj = uuid.UUID(user_id)
        except ValueError:
            logger.error(f"UUID inválido: {user_id}")
            raise HTTPException(status_code=400, detail="Invalid user_id format. Must be a valid UUID.")
        
        # First, load user data
        user_data = rag_service.load_user_data(user_id, user_name)
        form_of_address = user_data.get("user_data", {}).get("form_of_address", "")
        
        # Then generate augmented prompt
        system_prompt, augmented_prompt = rag_service.generate_user_augmented_prompt(
            user_id, user_name, form_of_address, query, k=k
        )
        return {
            "system_prompt": system_prompt,
            "augmented_prompt": augmented_prompt
        }
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Erro ao gerar prompt para o usuário: {str(e)}\n{error_trace}")
        return {"status": "error", "message": str(e)}

# JARVIS ASSISTANT ENDPOINTS

@app.post("/api/jarvis/ask")
async def jarvis_query(query: JarvisQueryRequest):
    """
    Endpoint para processar consultas sobre o JARVIS.
    
    Args:
        query (JarvisQueryRequest): Corpo da requisição contendo a query e histórico.
        
    Returns:
        JSONResponse: Resposta da consulta.
    """
    try:
        # Log para debug do histórico de conversa
        conversation_history = query.conversation_history or []
        
        # Logging detalhado do histórico para diagnóstico do problema
        logger.info(f"Received query: '{query.query}' with conversation history length: {len(conversation_history)}")
        for i, msg in enumerate(conversation_history):
            logger.info(f"History message {i}: role={msg.get('role', 'unknown')}, content preview={msg.get('content', '')[:30]}...")
        
        # Calcular número de pares de mensagens corretamente (importante para o estágio da conversa)
        # Um par completo consiste em uma mensagem do usuário seguida de uma resposta do assistente
        user_messages = sum(1 for msg in conversation_history if msg.get('role') == 'user')
        assistant_messages = sum(1 for msg in conversation_history if msg.get('role') == 'assistant')
        msg_count = min(user_messages, assistant_messages)
        
        # Se o último item for do usuário, estamos em um novo par incompleto
        if conversation_history and conversation_history[-1].get('role') == 'user':
            msg_count = min(user_messages - 1, assistant_messages)
        
        logger.info(f"Processing query with: {user_messages} user messages, {assistant_messages} assistant messages, {msg_count} complete message pairs")
        
        # Validar rapidamente o histórico de conversa
        if conversation_history:
            # Verificar e corrigir mensagens inválidas
            valid_conversation_history = []
            for i, message in enumerate(conversation_history):
                if "role" not in message or "content" not in message:
                    logger.warning(f"Invalid message at position {i} in conversation history: {message}")
                    continue
                if message["role"] not in ["user", "assistant"]:
                    logger.warning(f"Invalid role in message at position {i}: {message['role']}")
                    continue
                valid_conversation_history.append(message)
            
            conversation_history = valid_conversation_history
            
            # Verificar se os pares de mensagens são consistentes
            if len(conversation_history) >= 2:
                for i in range(0, len(conversation_history) - 1, 2):
                    if i + 1 < len(conversation_history):
                        if conversation_history[i]["role"] != "user" or conversation_history[i+1]["role"] != "assistant":
                            logger.warning(f"Inconsistent message pairs at positions {i} and {i+1}. Roles: {conversation_history[i]['role']}, {conversation_history[i+1]['role']}")
        
        # Gerar resposta com histórico validado
        response = jarvis_assistant.answer_query(
            query=query.query,
            conversation_history=conversation_history
        )
        
        # Adicionar estágio da conversa e config para debug
        config = jarvis_assistant.prompt_generator._adapt_to_conversation_stage(conversation_history)
        response["debug_conversation_stage"] = {
            "message_pairs": msg_count,
            "config": config
        }
        
        return JSONResponse(content=response)
    except Exception as e:
        logger.error(f"Error in JARVIS assistant query endpoint: {str(e)}", exc_info=True)
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={"error": "Erro interno ao processar sua consulta. Por favor, tente novamente mais tarde."}
        )

@app.post("/api/reload-jarvis-content")
async def reload_jarvis_content():
    """Reload the JARVIS information content from the text file"""
    global jarvis_assistant
    if not jarvis_assistant:
        logger.error("JARVIS Assistant service not initialized")
        raise HTTPException(status_code=500, detail="JARVIS Assistant service not initialized")
    
    try:
        # Reload JARVIS content
        result = jarvis_assistant.load_jarvis_content()
        return {
            "status": "success", 
            "message": f"JARVIS content reloaded successfully with {len(result['text'])} characters"
        }
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Erro ao recarregar conteúdo do JARVIS: {str(e)}\n{error_trace}")
        return {"status": "error", "message": str(e)} 