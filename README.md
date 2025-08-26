# Calum Worthy AI Assistant - Conversational AI System

A sophisticated AI chatbot system that creates an authentic digital twin of actor Calum Worthy, featuring advanced RAG (Retrieval-Augmented Generation), persistent memory, and conversational intelligence.

## 🎭 Overview

This system creates an AI avatar of Calum Worthy that can engage in natural conversations while maintaining context, personality, and memories across sessions. The AI responds authentically in Calum's voice, drawing from stored memories and conversation history to provide contextually rich interactions.

### Key Features

- **🧠 Persistent Memory**: Stores and retrieves conversation memories using DeepLake vector database
- **🎯 RAG System**: Retrieval-Augmented Generation for contextually aware responses
- **📝 Smart Summarization**: Automatic conversation compression with fact extraction
- **🎪 Authentic Persona**: Responds as Calum Worthy with his personality and speaking style
- **⚡ Dual Interface**: Both Streamlit web app and FastAPI REST endpoints
- **📊 LangSmith Integration**: Full conversation tracing and monitoring
- **🔄 Background Processing**: Non-blocking memory operations for fast responses

## 🏗️ System Architecture

### Core Components

#### 1. `convo.py` - The Conversation Engine
The heart of the system, implementing a stateful conversation graph with:

- **OrchestratedConversationalSystem**: Main orchestrator using LangGraph for conversation flow
- **DeepLakePDFStore**: Vector database interface for memory storage and retrieval
- **Smart Timeout Management**: Adaptive timeouts (25s initial, 10s standard) for reliable memory queries
- **Memory Management**: Automatic conversation summarization and fact extraction
- **Background Tasks**: Non-blocking memory storage and retrieval operations

**Key Functions:**
```python
async def turn(message: str) -> dict  # Main conversation handler
async def rag_query(query: str) -> List[str]  # Memory retrieval
async def add_memory(agent: str, text: str)  # Memory storage
```

#### 2. `fastapi_app.py` - REST API Interface
Production-ready FastAPI server providing:

- **POST /chat**: Main conversation endpoint
- **GET /health**: Health check endpoint
- **GET /disconnect**: Clean DeepLake disconnection
- **Enhanced Debugging**: Comprehensive RAG performance logging

#### 3. `streamlit_app.py` - Web Interface
Interactive web application with:

- Real-time chat interface
- Session management
- Memory visualization
- Configuration controls

#### 4. `llm_client.py` - Unified LLM Interface
Centralized OpenAI client with:

- LangSmith tracing integration
- Consistent model usage across components
- Error handling and retry logic

### Data Flow

```
User Input → FastAPI/Streamlit → Conversation System → Memory Retrieval (RAG) 
    ↓
System Prompt Construction ← Memory Context ← DeepLake Vector DB
    ↓
LLM Response Generation → Memory Storage (Background) → Response to User
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- ActiveLoop account for DeepLake
- LangSmith account (optional, for tracing)

### 1. Environment Setup

```bash
# Clone and navigate to project
cd legacy

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create `.streamlit/secrets.toml`:

```toml
# API Keys
OPENAI_API_KEY = "sk-your-openai-key"
ACTIVELOOP_TOKEN = "your-activeloop-token"
LANGSMITH_API_KEY = "lsv2_pt_your-langsmith-key"  # Optional

# Configuration
DATASET_NAME = "calum_v10"
PERSONA_PROMPT_FILE = "calum_prompt_v2.1.yaml"
ACTIVELOOP_ORG_ID = "your-org-id"

# LangSmith (Optional)
LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_PROJECT = "calum-worthy-chatbot-api"
```

### 3. Running the System

#### Option A: FastAPI (Recommended for API usage)

```bash
# Start the server
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload

# Test the API
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hi Calum, tell me about your acting career"}'
```

#### Option B: Streamlit (Web Interface)

```bash
# Start the web app
streamlit run streamlit_app.py
```

## 📡 API Reference

### POST /chat
Main conversation endpoint.

**Request:**
```json
{
  "message": "Your message to Calum",
  "thread_id": "optional-thread-id"
}
```

**Response:**
```json
{
  "response": "Calum's response",
  "thread_id": "conversation-thread-id",
  "facts_extracted": ["New facts learned"],
  "memories_used": 3
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-26T01:00:00Z",
  "dataset": "calum_v10",
  "prompt_file": "calum_prompt_v2.1.yaml"
}
```

### GET /disconnect
Gracefully disconnect from DeepLake.

## 🧠 Memory System

### How It Works

1. **Memory Storage**: Every conversation turn is embedded and stored in DeepLake
2. **Memory Retrieval**: Relevant memories are retrieved using semantic similarity
3. **Context Integration**: Retrieved memories are integrated into the system prompt
4. **Background Processing**: Memory operations happen asynchronously for speed

### Memory Types

- **User Messages**: What the user said
- **Assistant Responses**: Calum's replies
- **Extracted Facts**: Important information about the user
- **Compressed History**: Summarized conversation context

### RAG Performance

- **Initial Connection**: 25-second timeout for first DeepLake connection
- **Standard Queries**: 10-second timeout for subsequent queries
- **Background Completion**: Failed queries retry in background
- **Smart Timeout**: Adaptive based on connection state

## 🎯 Persona System

The AI maintains Calum's authentic personality through:

### Persona Configuration (`calum_prompt_v2.1.yaml`)

- **Identity**: Actor, activist, authentic personality
- **Speaking Style**: Casual, witty, engaging, never robotic
- **Interests**: Acting, method acting, social causes
- **Response Guidelines**: Natural conversation flow

### Memory-Informed Responses

The system uses retrieved memories to:
- Reference previous conversations
- Remember user preferences and details
- Maintain relationship continuity
- Provide contextually relevant responses

## 🛠️ Development Guide

### Key Configuration

```python
# Timeout Configuration
RESOLVE_CONTEXT_TIMEOUT = 30.0  # First query
RESOLVE_CONTEXT_STANDARD = 12.0  # Subsequent queries
RAG_QUERY_TIMEOUT = 25.0  # First query  
RAG_QUERY_STANDARD = 10.0  # Subsequent queries

# Memory Settings
TOP_K_MEMORIES = 3  # Number of memories to retrieve
SUMMARIZE_AFTER_TURNS = 12  # When to compress history
```

### Adding New Features

1. **New Endpoints**: Add to `fastapi_app.py`
2. **Conversation Logic**: Modify `convo.py` conversation graph
3. **Memory Processing**: Update `DeepLakePDFStore` methods
4. **UI Changes**: Edit `streamlit_app.py`

### Debugging

Enable debug mode for detailed logging:

```python
system = OrchestratedConversationalSystem(
    session_config=config,
    debug=True  # Enable detailed logging
)
```

Debug logs include:
- RAG query performance
- Memory retrieval details
- System prompt construction
- Timeout analysis

## 📊 Monitoring

### LangSmith Integration

- **Project**: `calum-worthy-chatbot-api`
- **Tracing**: Full conversation traces
- **Performance**: Response times and token usage
- **Debugging**: Error tracking and analysis

### Logs

The system provides comprehensive logging:

```
[INFO] ✅ Configuration loaded
[DEBUG] [DL] rag_query q='user question' k=3
[DEBUG] [NODE] Retrieved 3 memories in 2.5s
[INFO] Generated response: "Calum's reply..."
```

## 🧪 Testing

### API Tests

```bash
# Run test suite
python test_api.py

# Quick RAG test
python quick_test_rag.py
```

### Memory Tests

```bash
# Test memory storage and retrieval
python test_rag.py

# Test conversation flow
python test_complete_api.py
```

## 🔧 Troubleshooting

### Common Issues

1. **Memory Queries Timing Out**
   - Check DeepLake connection
   - Verify organization ID in dataset path
   - Increase timeout values if needed

2. **Empty Memory Responses**
   - Ensure dataset exists and has data
   - Check API keys and permissions
   - Verify memory storage is working

3. **Slow Initial Responses**
   - First query requires DeepLake connection setup (25-30s)
   - Subsequent queries should be faster (3-10s)
   - Consider connection warming strategies

### Performance Optimization

- **Memory Caching**: Implement local memory cache for frequent queries
- **Connection Pooling**: Maintain persistent DeepLake connections
- **Batch Processing**: Group memory operations for efficiency

## 📈 Scaling Considerations

### Production Deployment

- Use production ASGI server (Gunicorn + Uvicorn)
- Implement connection pooling
- Add rate limiting and authentication
- Monitor memory usage and query performance

### Database Scaling

- Consider DeepLake clustering for high throughput
- Implement memory archiving for long conversations
- Use read replicas for memory queries

## 🤝 Contributing

### Development Workflow

1. Create feature branch
2. Make changes with comprehensive tests
3. Update documentation
4. Submit pull request with performance analysis

### Code Standards

- Follow PEP 8 for Python code
- Include type hints
- Add comprehensive logging
- Write unit tests for new features

## 📝 License

This project is proprietary software developed for Calum Worthy's AI assistant system.

---

**Need Help?** Check the logs, run the test suites, or review the debug output for detailed system information.
