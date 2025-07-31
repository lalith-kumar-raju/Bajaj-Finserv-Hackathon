import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "your_pinecone_api_key_here")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "your_pinecone_environment_here")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "hackrx-documents")
    HACKRX_API_KEY = os.getenv("HACKRX_API_KEY", "031e2883dcfac08106d5a9982528deff7dcd207bd1efbca391476ea56fec65ac")
    
    # Model Settings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    LLM_MODEL = "gpt-4o"
    MAX_TOKENS = 4000
    TEMPERATURE = 0.1
    
    # Enhanced Document Processing
    CHUNK_SIZE = 800  # Optimized for better granularity
    CHUNK_OVERLAP = 150  # Optimized for better overlap
    MAX_CHUNKS_PER_DOCUMENT = 500  # Increased for complete document coverage
    
    # Enhanced Search Settings
    TOP_K_RESULTS = 10  # Increased from 5 for better retrieval
    SIMILARITY_THRESHOLD = 0.5  # Lowered from 0.7 for more results
    
    # API Settings
    API_TIMEOUT = 30
    MAX_CONCURRENT_REQUESTS = 10
    
    # Cache Settings
    ENABLE_CACHE = True
    CACHE_TTL = 3600  # 1 hour