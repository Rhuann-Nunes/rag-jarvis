import os
import json
from typing import Dict, List, Optional
from datetime import datetime
import pytz
import qdrant_client
from qdrant_client.http import models
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Qdrant
from supabase_client import SupabaseClient

load_dotenv()

class UserDataService:
    def __init__(self, use_in_memory=False, embedding_model=None):
        # Use provided embedding model or initialize one
        self.embedding_model = embedding_model or OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
        )
        
        self.collection_name = os.getenv("QDRANT_USER_COLLECTION_NAME", "user_data")
        self.supabase_client = SupabaseClient()
        
        # Initialize Brazil timezone
        self.brazil_tz = pytz.timezone('America/Sao_Paulo')
        
        # Initialize in-memory or external Qdrant client
        if use_in_memory:
            self.client = qdrant_client.QdrantClient(":memory:")
            self.vector_store = None
        else:
            self.client = qdrant_client.QdrantClient(
                url=os.getenv("QDRANT_HOST", "localhost"),
                port=int(os.getenv("QDRANT_PORT", 6333)),
                api_key=os.getenv("QDRANT_API_KEY")
            )
        
        # Create collection if it doesn't exist
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """Ensure the Qdrant collection exists, create it if it doesn't"""
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=1536,  # Dimension for text-embedding-3-small
                    distance=models.Distance.COSINE
                )
            )
    
    def get_current_datetime_br(self) -> str:
        """Get current datetime in Brazil timezone"""
        utc_now = datetime.now(pytz.UTC)
        br_now = utc_now.astimezone(self.brazil_tz)
        return br_now.strftime("%d/%m/%Y %H:%M")
    
    def _convert_to_documents(self, user_data: Dict) -> List[Document]:
        """Convert user data to documents for the vector store"""
        documents = []
        user_id = user_data.get("user_id", "")
        user_name = user_data.get("user_name", "")
        form_of_address = user_data.get("form_of_address", "")
        
        # Add current datetime information
        current_datetime = self.get_current_datetime_br()
        datetime_info = {
            "type": "datetime_info",
            "user_id": user_id,
            "current_datetime": current_datetime
        }
        datetime_text = f"Data e hora atual: {current_datetime}"
        documents.append(Document(page_content=datetime_text, metadata=datetime_info))
        
        # Add a document with basic user information
        base_info = {
            "type": "user_info",
            "user_id": user_id,
            "user_name": user_name,
            "form_of_address": form_of_address
        }
        base_info_text = f"Informações do usuário: Nome: {user_name}, Forma de tratamento: {form_of_address}"
        documents.append(Document(page_content=base_info_text, metadata=base_info))
        
        # Add projects as documents
        for project in user_data.get("projects", []):
            project_metadata = {
                "type": "project",
                "user_id": user_id,
                "project_id": project.get("id", ""),
                "project_name": project.get("name", ""),
                "project_color": project.get("color", ""),
                "created_at": project.get("created_at", ""),
                "updated_at": project.get("updated_at", "")
            }
            project_text = f"Projeto: {project.get('name', '')} com a cor {project.get('color', '')}"
            documents.append(Document(page_content=project_text, metadata=project_metadata))
        
        # Add tasks as documents
        for task in user_data.get("tasks", []):
            task_metadata = {
                "type": "task",
                "user_id": user_id,
                "task_id": task.get("id", ""),
                "task_title": task.get("title", ""),
                "task_description": task.get("description", ""),
                "task_completed": task.get("completed", False),
                "task_due_date": task.get("due_date", ""),
                "task_recurrence_type": task.get("recurrence_type", ""),
                "task_recurrence_interval": task.get("recurrence_interval", 0),
                "project_id": task.get("project_id", ""),
                "created_at": task.get("created_at", ""),
                "updated_at": task.get("updated_at", "")
            }
            
            due_date_str = ""
            if task.get("due_date"):
                try:
                    # Parse the UTC date
                    utc_date = datetime.fromisoformat(task.get("due_date").replace('Z', '+00:00'))
                    # Convert to Brazil timezone
                    br_date = utc_date.astimezone(self.brazil_tz)
                    # Format date and time in Brazilian format
                    due_date_str = br_date.strftime("%d/%m/%Y %H:%M")
                except Exception as e:
                    print(f"Erro ao converter data: {e}")
                    due_date_str = task.get("due_date", "")
            
            status = "Concluída" if task.get("completed", False) else "Pendente"
            task_text = f"Tarefa: {task.get('title', '')}. Descrição: {task.get('description', '')}. Status: {status}. "
            if due_date_str:
                task_text += f"Data de vencimento: {due_date_str}. "
            
            # Add detailed recurrence information
            recurrence_type = task.get("recurrence_type", "")
            recurrence_interval = task.get("recurrence_interval", 0)
            if recurrence_type and recurrence_interval:
                if recurrence_type == "daily":
                    task_text += f"Tarefa recorrente: Repete a cada {recurrence_interval} dia(s). "
                elif recurrence_type == "weekly":
                    task_text += f"Tarefa recorrente: Repete a cada {recurrence_interval} semana(s). "
                elif recurrence_type == "monthly":
                    task_text += f"Tarefa recorrente: Repete a cada {recurrence_interval} mês(es). "
                elif recurrence_type == "yearly":
                    task_text += f"Tarefa recorrente: Repete a cada {recurrence_interval} ano(s). "
                else:
                    task_text += f"Tipo de recorrência: {recurrence_type}. Intervalo de recorrência: {recurrence_interval}. "
            
            documents.append(Document(page_content=task_text, metadata=task_metadata))
        
        # Add user preferences
        preferences = user_data.get("preferences", {})
        if preferences:
            pref_metadata = {
                "type": "preferences",
                "user_id": user_id,
                "form_of_address": preferences.get("form_of_address", ""),
                "phone_number": preferences.get("phone_number", ""),
                "allow_notifications": preferences.get("allow_notifications", False)
            }
            pref_text = f"Preferências do usuário: Forma de tratamento: {preferences.get('form_of_address', '')}. "
            pref_text += f"Telefone: {preferences.get('phone_number', '')}. "
            pref_text += f"Permite notificações: {'Sim' if preferences.get('allow_notifications', False) else 'Não'}."
            
            documents.append(Document(page_content=pref_text, metadata=pref_metadata))
        
        return documents
    
    def load_user_data(self, user_id: str, user_name: str = "") -> Dict:
        """Load user data from Supabase and store in vector database"""
        # Get user data from Supabase
        user_data = self.supabase_client.get_all_user_data(user_id, user_name)
        
        # Convert to documents
        documents = self._convert_to_documents(user_data)
        
        # Create a new vector store with the documents
        self.vector_store = Qdrant.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            location=":memory:",
            collection_name=self.collection_name
        )
        
        return user_data
    
    def search_user_data(self, user_id: str, query: str, limit: int = 5) -> List[Dict]:
        """Search user data with the query"""
        if not self.vector_store:
            return []
            
        # Search in vector store
        results = self.vector_store.similarity_search_with_score(query, k=limit)
        
        # Format results
        formatted_results = []
        for doc, score in results:
            if doc.metadata.get("user_id") == user_id:
                formatted_results.append({
                    "score": float(score),
                    "text": doc.page_content,
                    "metadata": doc.metadata
                })
                
        return formatted_results 