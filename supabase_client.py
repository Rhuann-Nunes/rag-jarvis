import os
from typing import Dict, List, Optional
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

class SupabaseClient:
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_SERVICE_KEY")  # Using the service role key for more access
        self.client: Client = create_client(self.url, self.key)
    
    def get_user_projects(self, user_id: str) -> List[Dict]:
        """Get all projects for a user from the projects table (public schema)"""
        try:
            response = self.client.table("projects").select("*").eq("user_id", user_id).execute()
            return response.data
        except Exception as e:
            print(f"Error getting user projects: {e}")
            return []
    
    def get_user_tasks(self, user_id: str) -> List[Dict]:
        """Get all tasks for a user from the tasks table (public schema)"""
        try:
            response = self.client.table("tasks").select("*").eq("user_id", user_id).execute()
            return response.data
        except Exception as e:
            print(f"Error getting user tasks: {e}")
            return []
    
    def get_user_preferences(self, user_id: str) -> Optional[Dict]:
        """Get user preferences from the user_preferences table (public schema)"""
        try:
            response = self.client.table("user_preferences").select("*").eq("user_id", user_id).execute()
            user_preferences = response.data
            if user_preferences and len(user_preferences) > 0:
                return user_preferences[0]
            return None
        except Exception as e:
            print(f"Error getting user preferences: {e}")
            return None
    
    def get_all_user_data(self, user_id: str, user_name: str = "") -> Dict:
        """Get all relevant data for a user from multiple tables"""
        user_projects = self.get_user_projects(user_id)
        user_tasks = self.get_user_tasks(user_id)
        user_preferences = self.get_user_preferences(user_id)
        
        form_of_address = ""
        
        # Get the preferred form of address
        if user_preferences and "form_of_address" in user_preferences:
            form_of_address = user_preferences.get("form_of_address", "")
        
        return {
            "user_id": user_id,
            "user_name": user_name,
            "form_of_address": form_of_address,
            "projects": user_projects,
            "tasks": user_tasks,
            "preferences": user_preferences
        } 