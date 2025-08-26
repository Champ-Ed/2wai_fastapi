import os
import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, Optional, List, Any
# Updated: Fixed syntax error in convo.py
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler with formatting
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# Load environment variables and secrets
def load_environment_variables():
    """Load environment variables and secrets from .streamlit/secrets.toml or environment."""
    try:
        # Try to load from Streamlit secrets.toml file for consistency
        import toml
        secrets_path = os.path.join(os.path.dirname(__file__), ".streamlit", "secrets.toml")
        
        if os.path.exists(secrets_path):
            secrets = toml.load(secrets_path)
            api_key = secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
            activeloop_token = secrets.get("ACTIVELOOP_TOKEN", os.getenv("ACTIVELOOP_TOKEN", ""))
            activeloop_org = secrets.get("ACTIVELOOP_ORG_ID", os.getenv("ACTIVELOOP_ORG_ID", ""))
            langchain_key = secrets.get("LANGCHAIN_API_KEY", os.getenv("LANGCHAIN_API_KEY", ""))
            dataset_name = secrets.get("DATASET_NAME", os.getenv("DATASET_NAME", "calum_v10"))
            persona_prompt_file = secrets.get("PERSONA_PROMPT_FILE", os.getenv("PERSONA_PROMPT_FILE", "calum_prompt_v2.1.yaml"))
            logger.info("✅ Loaded configuration from .streamlit/secrets.toml")
        else:
            # Fallback to environment variables
            api_key = os.getenv("OPENAI_API_KEY", "")
            activeloop_token = os.getenv("ACTIVELOOP_TOKEN", "")
            activeloop_org = os.getenv("ACTIVELOOP_ORG_ID", "")
            langchain_key = os.getenv("LANGCHAIN_API_KEY", "")
            dataset_name = os.getenv("DATASET_NAME", "calum_v10")
            persona_prompt_file = os.getenv("PERSONA_PROMPT_FILE", "calum_prompt_v2.1.yaml")
            logger.info("⚠️ Using environment variables (secrets.toml not found)")
    except ImportError:
        logger.warning("toml library not available, using environment variables only")
        # Fallback to environment variables when toml not available
        api_key = os.getenv("OPENAI_API_KEY", "")
        activeloop_token = os.getenv("ACTIVELOOP_TOKEN", "")
        activeloop_org = os.getenv("ACTIVELOOP_ORG_ID", "")
        langchain_key = os.getenv("LANGCHAIN_API_KEY", "")
        dataset_name = os.getenv("DATASET_NAME", "calum_v10")
        persona_prompt_file = os.getenv("PERSONA_PROMPT_FILE", "calum_prompt_v2.1.yaml")
    except Exception as e:
        logger.error(f"Error loading secrets: {e}")
        # Final fallback to environment variables
        api_key = os.getenv("OPENAI_API_KEY", "")
        activeloop_token = os.getenv("ACTIVELOOP_TOKEN", "")
        activeloop_org = os.getenv("ACTIVELOOP_ORG_ID", "")
        langchain_key = os.getenv("LANGCHAIN_API_KEY", "")
        dataset_name = os.getenv("DATASET_NAME", "calum_v10")
        persona_prompt_file = os.getenv("PERSONA_PROMPT_FILE", "calum_prompt_v2.1.yaml")

    # Set up LangSmith tracing BEFORE importing conversation system
    os.environ.setdefault("LANGCHAIN_API_KEY", langchain_key)
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "calum-worthy-chatbot-api")
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    os.environ.setdefault("OPENAI_API_KEY", api_key)
    os.environ.setdefault("ACTIVELOOP_TOKEN", activeloop_token)
    os.environ.setdefault("ACTIVELOOP_ORG_ID", activeloop_org)
    
    # Initialize unified LLM client with LangSmith tracing
    try:
        from llm_client import initialize_client
        initialize_client(
            api_key=api_key,
            base_url="https://api.openai.com/v1", 
            enable_tracing=bool(langchain_key)  # Enable tracing only if we have the key
        )
        logger.info("✅ Unified LLM client initialized with LangSmith tracing")
    except Exception as e:
        logger.error(f"❌ Failed to initialize unified LLM client: {e}")
    
    # Debug configuration
    if langchain_key:
        logger.info(f"✅ LangSmith configured with API key: {langchain_key[:8]}...")
        logger.info(f"✅ LangSmith project: calum-worthy-chatbot-api")
    else:
        logger.warning("⚠️ LangSmith API key not found - tracing disabled")
    
    logger.info(f"✅ Dataset configured: {dataset_name}")
    logger.info(f"✅ Prompt file configured: {persona_prompt_file}")
    
    return api_key, activeloop_token, activeloop_org, langchain_key, dataset_name, persona_prompt_file

# Load environment variables at startup
api_key, activeloop_token, activeloop_org, langchain_key, dataset_name, persona_prompt_file = load_environment_variables()

# Verify LangSmith tracing is enabled
def verify_langsmith():
    """Verify LangSmith tracing setup"""
    try:
        from langsmith import Client
        if langchain_key:
            client = Client()
            logger.info("✅ LangSmith client initialized successfully")
            return True
        else:
            logger.warning("⚠️ No LangSmith API key - tracing disabled")
            return False
    except Exception as e:
        logger.error(f"❌ LangSmith setup failed: {e}")
        return False

# Initialize LangSmith
langsmith_enabled = verify_langsmith()

# Pydantic models for API requests/responses
class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's message to send to Calum")
    thread_id: Optional[str] = Field(None, description="Optional thread ID for conversation continuity")
    user_id: Optional[str] = Field(None, description="Optional user ID for personalization")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Calum's response message")
    thread_id: str = Field(..., description="Thread ID for this conversation")
    timestamp: str = Field(..., description="Timestamp of the response")
    facts_extracted: Optional[List[str]] = Field(None, description="Any new facts extracted about the user")
    turn_count: Optional[int] = Field(None, description="Number of conversation turns in this thread")
    context_used: Optional[str] = Field(None, description="Context retrieved from memory for this response")

class ConversationInfo(BaseModel):
    thread_id: str
    created_at: str
    last_updated: str
    message_count: int
    summary: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

# Global conversation systems cache
conversation_systems: Dict[str, Any] = {}

# Global embedding model cache
_embedding_model = None
_conversation_system_class = None

def get_embedding_model():
    """Lazy load the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding
            from llama_index.core import Settings
            
            _embedding_model = OpenAIEmbedding(
                model="text-embedding-3-small",
                api_key=api_key
            )
            Settings.embed_model = _embedding_model
            logger.info("Initialized OpenAI embedding model")
        except ImportError as e:
            logger.error(f"Failed to import llama_index: {e}")
            raise
    
    return _embedding_model

def get_conversation_system_class():
    """Lazy load the conversation system class."""
    global _conversation_system_class
    if _conversation_system_class is None:
        try:
            # Mock Streamlit to allow importing convo.py in FastAPI
            import sys
            from types import ModuleType
            
            # Create a mock streamlit module if it's not available or causes issues
            if 'streamlit' not in sys.modules:
                mock_st = ModuleType('streamlit')
                mock_st.secrets = {}  # Mock secrets
                sys.modules['streamlit'] = mock_st
            
            # Import the original conversation system from convo.py
            from convo import OrchestratedConversationalSystem
            _conversation_system_class = OrchestratedConversationalSystem
            logger.info("SUCCESS: Loaded OrchestratedConversationalSystem from convo.py - same as Streamlit!")
            
        except ImportError as e:
            logger.error(f"Failed to import original conversation system: {e}")
            # Fallback to complete system
            try:
                from complete_calum import CompleteCalumSystem
                _conversation_system_class = CompleteCalumSystem
                logger.info("Using CompleteCalumSystem as fallback")
            except ImportError:
                # Final fallback to mock system
                class MockConversationSystem:
                    def __init__(self, session):
                        self.session = session
                    
                    async def run_turn_fast(self, user_input):
                        return {
                            "response": f"Hello! You said: '{user_input}'. This is a test response from the API.",
                            "extracted_facts": [],
                            "thread_id": self.session.get("thread_id", "test-thread")
                        }
                    
                    async def _cleanup(self):
                        pass
                
                _conversation_system_class = MockConversationSystem
                logger.info("Using mock conversation system as final fallback")
        except Exception as e:
            logger.error(f"Unexpected error importing conversation system: {e}")
            # Final fallback
            class MockConversationSystem:
                def __init__(self, session):
                    self.session = session
                
                async def run_turn_fast(self, user_input):
                    return {
                        "response": f"Hello! You said: '{user_input}'. This is a test response from the API.",
                        "extracted_facts": [],
                        "thread_id": self.session.get("thread_id", "test-thread")
                    }
                
                async def _cleanup(self):
                    pass
            
            _conversation_system_class = MockConversationSystem
            logger.info("Using mock conversation system due to unexpected error")
    
    return _conversation_system_class

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown events."""
    # Startup
    logger.info("Starting Calum Worthy AI Assistant API...")
    
    # Don't initialize heavy dependencies at startup - do it lazily
    logger.info("API startup complete!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Calum Worthy AI Assistant API...")
    
    # Clean up all conversation systems
    for thread_id, conv_system in conversation_systems.items():
        try:
            if hasattr(conv_system, '_cleanup'):
                await conv_system._cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up conversation {thread_id}: {e}")
    
    conversation_systems.clear()
    logger.info("API shutdown complete!")

# Initialize FastAPI app
app = FastAPI(
    title="Calum Worthy AI Assistant API",
    description="REST API for Calum Worthy's AI conversational system",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _resolve_db_path() -> str:
    val = os.environ.get("LANGGRAPH_CHECKPOINT_DB", "api_checkpoints.sqlite")
    
    if "://" in val:
        return val

    if not os.path.isabs(val):
        return os.path.join(os.path.dirname(__file__), val)

    return val

async def get_conversation_system(thread_id: str):
    """Get or create a conversation system for the given thread ID - same as Streamlit!"""
    if thread_id not in conversation_systems:
        # Configure database path (same as Streamlit)
        _db_path = _resolve_db_path()
        os.environ["LANGGRAPH_CHECKPOINT_DB"] = _db_path
        
        # Session configuration - EXACT SAME as streamlit_app.py but with configurable values
        session = {
            "api_key": api_key,
            "model_name": "gpt-4o",
            "base_url": "https://api.openai.com/v1",
            "persona_name": "Calum",
            "avatar_id": "calum",
            "avatar_prompts": {
                "calum": "You are Calum Worthy, a witty activist and actor."
            },
            "temperature": 0.3,
            "debug": True,  # Enable debug to see checkpoint behavior
            "force_sync_flush": False,
            "thread_id": thread_id,
            "checkpoint_db": _db_path,  # pass explicit path to backend
            "summarize_after_turns": 12,       # Summarize when chat reaches 12 turns
            "turns_to_keep_after_summary": 4,  # Keep last 4 turns after summarizing
            "enable_tracing": bool(langchain_key),  # Pass LangSmith configuration
            "dataset_name": dataset_name,  # Configurable dataset name
            "persona_prompt_path": persona_prompt_file,  # Configurable prompt file
            "activeloop_org_id": activeloop_org  # Pass organization ID for DeepLake
        }
        
        # Initialize the embedding model BEFORE constructing the conversation system (same as Streamlit)
        get_embedding_model()
        
        # Create conversation system - same as Streamlit
        ConversationSystemClass = get_conversation_system_class()
        conv_system = ConversationSystemClass(session=session)
        conversation_systems[thread_id] = conv_system
        
        logger.info(f"Created new conversation system for thread: {thread_id} (same config as Streamlit)")
    
    return conversation_systems[thread_id]

def generate_thread_id() -> str:
    """Generate a new thread ID."""
    return str(uuid.uuid4())

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a message to Calum and get a response."""
    try:
        # Generate thread ID if not provided
        thread_id = request.thread_id or generate_thread_id()
        
        # Get conversation system for this thread
        conv_system = await get_conversation_system(thread_id)
        
        # Process the message
        logger.info(f"Processing message for thread {thread_id}: {request.message[:100]}...")
        
        # Run the conversation turn
        logger.info(f"Running conversation turn for thread {thread_id}")
        result_state = await conv_system.run_turn_fast(request.message)
        logger.info(f"Conversation turn completed. Result keys: {list(result_state.keys())}")
        
        # Extract response and any facts
        response_text = result_state.get("response", "Sorry, I had trouble processing that. Could you try again?")
        extracted_facts = result_state.get("extracted_facts", [])
        
        # Debug RAG context more thoroughly
        selected_context = result_state.get("selected_context", "")
        if selected_context and "Memories:" in selected_context:
            logger.info(f"RAG memories retrieved: {selected_context[:200]}...")
        else:
            logger.warning(f"No memories in selected_context: {selected_context}")
            
        # Check if system prompt contains context
        agent_context = result_state.get("agent_context", "")
        if agent_context and ("Memories:" in agent_context or "Context:" in agent_context):
            logger.info(f"System prompt contains RAG context (length: {len(agent_context)})")
        else:
            logger.warning("System prompt may be missing RAG context")
            
        # Check if memories were used
        if "memories" in result_state:
            logger.info(f"Memories used: {len(result_state.get('memories', []))} items")
        else:
            logger.warning("No memories found in result_state")
        
        logger.info(f"Generated response for thread {thread_id}: {response_text[:100]}...")
        
        return ChatResponse(
            response=response_text,
            thread_id=thread_id,
            timestamp=datetime.now().isoformat(),
            facts_extracted=extracted_facts[-5:] if extracted_facts else None,  # Return last 5 facts
            context_used=result_state.get("selected_context", "No context retrieved")[:500]  # Include selected_context instead
        )
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/conversations/{thread_id}", response_model=ConversationInfo)
async def get_conversation_info(thread_id: str):
    """Get information about a specific conversation thread."""
    try:
        if thread_id not in conversation_systems:
            raise HTTPException(status_code=404, detail="Conversation thread not found")
        
        conv_system = conversation_systems[thread_id]
        
        # Get conversation state (this would need to be implemented in the conversation system)
        # For now, return basic info
        return ConversationInfo(
            thread_id=thread_id,
            created_at=datetime.now().isoformat(),  # This should come from actual creation time
            last_updated=datetime.now().isoformat(),
            message_count=0,  # This should come from actual message count
            summary=None  # This should come from compressed_history if available
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation info: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.delete("/conversations/{thread_id}")
async def delete_conversation(thread_id: str):
    """Delete a conversation thread and properly disconnect from DeepLake."""
    try:
        if thread_id in conversation_systems:
            # Get the conversation system
            conv_system = conversation_systems[thread_id]
            
            # Properly cleanup the conversation system (closes DeepLake connections)
            try:
                if hasattr(conv_system, 'store') and hasattr(conv_system.store, 'cleanup'):
                    await conv_system.store.cleanup()
                if hasattr(conv_system, '_cleanup'):
                    await conv_system._cleanup()
            except Exception as cleanup_error:
                logger.warning(f"Error during conversation cleanup: {cleanup_error}")
            
            # Remove from cache
            del conversation_systems[thread_id]
            
            logger.info(f"Deleted and properly disconnected conversation thread: {thread_id}")
            return {"message": f"Conversation {thread_id} deleted and disconnected successfully"}
        else:
            raise HTTPException(status_code=404, detail="Conversation thread not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/conversations/{thread_id}/disconnect")
async def disconnect_conversation(thread_id: str):
    """Disconnect from DeepLake for a specific conversation without deleting it."""
    try:
        if thread_id in conversation_systems:
            conv_system = conversation_systems[thread_id]
            
            # Disconnect from DeepLake but keep the conversation system
            try:
                if hasattr(conv_system, 'store') and hasattr(conv_system.store, 'cleanup'):
                    await conv_system.store.cleanup()
                    logger.info(f"Disconnected from DeepLake for thread: {thread_id}")
            except Exception as cleanup_error:
                logger.warning(f"Error during DeepLake disconnection: {cleanup_error}")
            
            return {"message": f"Disconnected from DeepLake for conversation {thread_id}"}
        else:
            raise HTTPException(status_code=404, detail="Conversation thread not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error disconnecting conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/disconnect-all")
async def disconnect_all_conversations():
    """Disconnect from DeepLake for all active conversations."""
    try:
        disconnected_count = 0
        
        for thread_id, conv_system in conversation_systems.items():
            try:
                if hasattr(conv_system, 'store') and hasattr(conv_system.store, 'cleanup'):
                    await conv_system.store.cleanup()
                    disconnected_count += 1
            except Exception as cleanup_error:
                logger.warning(f"Error disconnecting thread {thread_id}: {cleanup_error}")
        
        logger.info(f"Disconnected from DeepLake for {disconnected_count} conversations")
        return {"message": f"Disconnected from DeepLake for {disconnected_count} conversations"}
        
    except Exception as e:
        logger.error(f"Error disconnecting all conversations: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/conversations", response_model=List[ConversationInfo])
async def list_conversations():
    """List all active conversation threads."""
    try:
        conversations = []
        for thread_id in conversation_systems.keys():
            conversations.append(ConversationInfo(
                thread_id=thread_id,
                created_at=datetime.now().isoformat(),  # This should come from actual creation time
                last_updated=datetime.now().isoformat(),
                message_count=0,  # This should come from actual message count
                summary=None
            ))
        
        return conversations
        
    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Run the FastAPI app
    uvicorn.run(
        "fastapi_app:app",
        host="127.0.0.1",  # Use localhost instead of 0.0.0.0
        port=8000,
        reload=False,  # Disable reload to prevent restart issues
        log_level="info"
    )
