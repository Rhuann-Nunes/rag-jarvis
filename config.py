import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Groq Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "deepseek-r1-distill-llama-70b")

# Qdrant Configuration - now used only for collection naming
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "mistral_papers")
QDRANT_USER_COLLECTION_NAME = os.getenv("QDRANT_USER_COLLECTION_NAME", "user_data")

# Embedding Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Check if required environment variables are set
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable must be set")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable must be set (for embeddings)")
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise ValueError("Supabase configuration must be set (SUPABASE_URL and SUPABASE_SERVICE_KEY)") 