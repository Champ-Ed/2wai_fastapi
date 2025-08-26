# Calum Worthy AI Assistant - Conversational AI System

A sophisticated AI chatbot system that creates an authentic digital twin of actor Calum Worthy, featuring advanced RAG (Retrieval-Augmented Generation), persistent memory, and conversational intelligence.

## 🎭 Overview

This system creates an AI avatar of Calum Worthy that can engage in natural conversations while maintaining context, personality, and memories across sessions. The AI responds authentically in Calum's voice, drawing from stored memories and conversation history to provide contextually rich interactions.

## 💬 Conversation Intelligence

### What Makes This Different

Unlike simple chatbots, this system creates **genuine conversational relationships**:

#### **Contextual Memory Integration**
- **Semantic Understanding**: Finds relevant context even when topics are discussed differently
- **Relationship Building**: Conversations naturally evolve and deepen over time
- **Emotional Continuity**: Maintains awareness of previous emotional contexts and relationship dynamics
- **Topic Threading**: Connects related conversations across multiple sessions

#### **Authentic Persona Modeling**
- **Natural Speech Patterns**: Calum's authentic voice, wit, and conversational style
- **Adaptive Familiarity**: Adjusts intimacy level based on conversation history
- **Contextual Reactions**: Responds based on what you've shared and discussed together
- **Genuine Interest**: Asks follow-up questions and remembers what matters to you

#### **Progressive Relationship Development**
```
First Meeting → Getting Acquainted → Shared Interests → Deeper Conversations → Ongoing Friendship
```

Each conversation builds on previous interactions, creating an evolving relationship that feels authentic and meaningful.

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
- **Intelligent Context Building**: Dynamically retrieves and integrates relevant memories into each conversation turn
- **Conversational Memory**: Maintains long-term relationships by remembering user preferences, details, and conversation history
- **Adaptive RAG Pipeline**: Semantically searches through conversation history to provide contextually relevant responses

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

#### 5. `ingestion_deeplake.py` - Standalone file to upload pdf/text data files to deeplake


### Data Flow

```
User Input → FastAPI/Streamlit → Conversation Analysis → Semantic Memory Search
    ↓
Retrieved Context + User History → Intelligent Prompt Construction → Calum's Personality Layer
    ↓
LLM Response Generation → Contextual Response + Relationship Memory Updates → User
```

**Detailed Conversation Flow:**

1. **Input Analysis**: User message analyzed for intent, emotional tone, and topic
2. **Memory Retrieval**: Semantic search finds 3-5 most relevant previous conversation segments
3. **Context Building**: Retrieved memories integrated with current conversation state
4. **Persona Application**: Calum's personality and speaking style applied to context
5. **Response Generation**: LLM creates response with full conversational awareness
6. **Memory Updates**: New conversation turn stored for future context retrieval

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

### Conversational Intelligence

The system maintains sophisticated conversational intelligence through a multi-layered memory approach:

#### **Context Retrieval & Integration**
1. **Semantic Memory Search**: When you ask a question, the system searches through all previous conversations using semantic similarity to find the most relevant context
2. **Dynamic Context Building**: Retrieved memories are intelligently integrated into the conversation prompt, ensuring Calum responds with full awareness of your history together
3. **Relationship Continuity**: The AI remembers not just facts, but the flow and tone of your conversations, maintaining authentic relationship dynamics

#### **Multi-Type Memory Storage**
- **Conversational Turns**: Complete user messages and Calum's responses preserved with semantic embeddings
- **Extracted Facts**: Important personal details about users automatically identified and stored
- **Compressed Summaries**: Long conversation histories intelligently summarized while preserving key emotional and factual content
- **Context Threads**: Related conversation topics linked together for deeper contextual understanding

#### **RAG (Retrieval-Augmented Generation) Pipeline**

The RAG system powers Calum's contextual awareness:

```
User Question → Semantic Search → Memory Retrieval → Context Integration → Informed Response
```

**How RAG Enhances Conversations:**
- **Personalized Responses**: Calum references specific things you've discussed before
- **Emotional Continuity**: Maintains awareness of your relationship dynamics and previous emotional context
- **Topic Threading**: Connects current questions to related past conversations
- **Progressive Understanding**: Builds deeper knowledge about you over time

#### **Example RAG in Action**

**You:** "How's your acting going?"
**System Process:**
1. Searches memories for: acting projects, career updates, method acting discussions
2. Retrieves relevant context: Previous conversations about specific roles, acting techniques discussed
3. Integrates context: Builds system prompt with your history of discussing acting together
4. **Calum responds:** "Hey! Actually, since we last talked about my method acting approach, I've been working on this really intense drama where..." *(references your specific previous conversation)*

### Memory Performance
- **Retrieval Speed**: 3-10 seconds for contextual memory integration
- **Semantic Accuracy**: Finds relevant memories even with different phrasing
- **Context Depth**: Typically retrieves 3-5 most relevant conversation segments
- **Background Processing**: New memories stored automatically without interrupting conversation flow

## 🎯 Persona System

The AI maintains Calum's authentic personality through sophisticated conversation modeling:

### Persona Configuration (`calum_prompt_v2.1.yaml`)

- **Identity**: Actor, activist, authentic personality with genuine speaking patterns
- **Speaking Style**: Casual, witty, engaging, never robotic - designed to feel like chatting with the real Calum
- **Interests & Expertise**: Acting techniques, method acting, social causes, entertainment industry insights
- **Conversation Guidelines**: Natural flow, authentic reactions, contextual awareness

### Memory-Informed Conversations

The system uses retrieved memories to create authentic relationship dynamics:

#### **Contextual Awareness**
- **References Previous Conversations**: "Remember when we talked about..." style natural callbacks
- **Builds on Shared History**: Conversations evolve and deepen over time based on what you've discussed
- **Maintains Relationship Tone**: Whether you're new friends or have chatted many times, Calum adjusts his familiarity level
- **Topic Continuity**: Picks up threads from previous conversations naturally

#### **Conversation Intelligence**
- **Emotional Memory**: Remembers the emotional context of previous interactions
- **Interest Tracking**: Recalls what topics you're passionate about and brings them up naturally
- **Personal Details**: Integrates facts about your life into relevant conversation moments
- **Conversational Growth**: The relationship develops authentically as you chat more

#### **Example Conversation Evolution**

**First Chat:**
- **You:** "Hi Calum!"
- **Calum:** "Hey there! Nice to meet you. What brings you my way today?"

**After Several Conversations:**
- **You:** "Hi Calum!"
- **Calum:** "Hey! Good to see you again. How's that project you were telling me about going? Last time you mentioned you were really excited about the creative direction..."

*Notice how the second response shows relationship memory, topic continuity, and authentic familiarity growth.*

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



