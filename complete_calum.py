"""
Complete Calum conversation system for FastAPI with RAG, summarization, and context management.
Standalone implementation without complex dependencies.
"""
import os
import uuid
import json
import asyncio
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
import logging

try:
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

class CompleteCalumSystem:
    """Complete Calum conversation system with RAG, summarization, and context management."""
    
    def __init__(self, session: Dict):
        self.session = session
        self.api_key = session.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.model_name = session.get("model_name", "gpt-4o")
        self.thread_id = session.get("thread_id", str(uuid.uuid4()))
        self.temperature = session.get("temperature", 0.3)
        self.summarize_after_turns = session.get("summarize_after_turns", 12)
        self.turns_to_keep_after_summary = session.get("turns_to_keep_after_summary", 4)
        
        # Load Calum's persona
        self.persona_prompt = self._load_persona()
        
        # Initialize LLM client
        self.llm_client = self._init_llm()
        
        # Initialize database for conversation storage
        self.db_path = f"conversations_{self.thread_id}.db"
        self._init_database()
        
        # Memory stores
        self.conversation_history = []
        self.extracted_facts = []
        self.compressed_history = ""
        self.turn_count = 0
        
        # Load existing conversation
        self._load_conversation()
    
    def _load_persona(self) -> str:
        """Load Calum's persona from YAML file."""
        # Try both potential YAML files
        for filename in ["calum_prompt_v2.1.yaml", "calum_twitter_v1.1.yaml"]:
            persona_path = Path(filename)
            if persona_path.exists():
                try:
                    with open(persona_path, 'r', encoding='utf-8') as f:
                        persona_data = yaml.safe_load(f)
                        # Try different possible keys
                        for key in ['calum_persona', 'SystemPrompt', 'content']:
                            if key in persona_data:
                                content = persona_data[key]
                                if isinstance(content, dict) and 'content' in content:
                                    return content['content']
                                elif isinstance(content, str):
                                    return content
                except Exception as e:
                    logger.warning(f"Error loading persona from {filename}: {e}")
        
        return self._default_persona()
    
    def _default_persona(self) -> str:
        """Comprehensive default Calum persona."""
        return """You are Calum Worthy, a witty Canadian actor and activist. You're known for:

PERSONALITY:
- Being genuine, warm, and slightly quirky in conversation
- Your work as an actor (Austin & Ally, The Act, etc.)
- Your activism and social awareness, especially climate issues
- A good sense of humor with occasional dad jokes
- Being thoughtful and empathetic
- Speaking in a natural, conversational way
- Using contractions and casual language
- Being emotionally intelligent and matching tone appropriately

CONVERSATION STYLE:
- Keep responses engaging but not too long
- Show your personality while being helpful and authentic
- Remember conversations and build relationships with users over time
- Avoid generic, overly-enthusiastic responses
- Be specific and earned in compliments
- Use natural, casual language with words like "honestly," "tbh," "like"
- Use emojis where they feel natural but don't overdo it

EXPERTISE:
- Hollywood history and acting (both comedy and drama)
- Climate crisis (from a scientific/activist view)
- AI and technology
- Avoid partisan politics but can discuss climate solutions

Remember previous conversations and facts about users to build meaningful relationships."""
    
    def _init_llm(self):
        """Initialize LLM client with available library."""
        if LANGCHAIN_AVAILABLE:
            return ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                temperature=self.temperature
            )
        elif OPENAI_AVAILABLE:
            return OpenAI(api_key=self.api_key)
        else:
            return None
    
    def _init_database(self):
        """Initialize SQLite database for conversation storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                turn_number INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                source_message TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_conversation(self):
        """Load existing conversation from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Load messages
        cursor.execute('SELECT role, content, timestamp, turn_number FROM messages ORDER BY id')
        messages = cursor.fetchall()
        
        for role, content, timestamp, turn_number in messages:
            self.conversation_history.append({
                "role": role,
                "content": content,
                "timestamp": timestamp,
                "turn_number": turn_number
            })
            if turn_number:
                self.turn_count = max(self.turn_count, turn_number)
        
        # Load facts
        cursor.execute('SELECT fact FROM facts ORDER BY id')
        facts = cursor.fetchall()
        self.extracted_facts = [fact[0] for fact in facts]
        
        # Load compressed history
        cursor.execute('SELECT value FROM metadata WHERE key = "compressed_history"')
        result = cursor.fetchone()
        if result:
            self.compressed_history = result[0]
        
        conn.close()
    
    def _save_message(self, role: str, content: str, turn_number: int = None):
        """Save message to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO messages (role, content, timestamp, turn_number)
            VALUES (?, ?, ?, ?)
        ''', (role, content, datetime.now().isoformat(), turn_number))
        
        conn.commit()
        conn.close()
    
    def _save_facts(self, new_facts: List[str], source_message: str):
        """Save new facts to database."""
        if not new_facts:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for fact in new_facts:
            cursor.execute('''
                INSERT INTO facts (fact, timestamp, source_message)
                VALUES (?, ?, ?)
            ''', (fact, datetime.now().isoformat(), source_message))
        
        conn.commit()
        conn.close()
    
    def _save_compressed_history(self, compressed_history: str):
        """Save compressed history to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO metadata (key, value)
            VALUES ("compressed_history", ?)
        ''', (compressed_history,))
        
        conn.commit()
        conn.close()
    
    async def run_turn_fast(self, user_input: str) -> Dict:
        """Process a conversation turn with full RAG and summarization."""
        try:
            self.turn_count += 1
            
            # Save user input
            self.conversation_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat(),
                "turn_number": self.turn_count
            })
            self._save_message("user", user_input, self.turn_count)
            
            # Query memory/context (RAG simulation)
            context = await self._query_memory(user_input)
            
            # Generate response with full context
            response = await self._generate_response_with_context(user_input, context)
            
            # Save response
            self.conversation_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().isoformat(),
                "turn_number": self.turn_count
            })
            self._save_message("assistant", response, self.turn_count)
            
            # Extract facts
            new_facts = await self._extract_facts(user_input, response)
            if new_facts:
                self.extracted_facts.extend(new_facts)
                self._save_facts(new_facts, user_input)
            
            # Check if we need to summarize
            if self.turn_count % self.summarize_after_turns == 0:
                await self._summarize_conversation()
            
            return {
                "response": response,
                "thread_id": self.thread_id,
                "extracted_facts": new_facts,
                "timestamp": datetime.now().isoformat(),
                "turn_count": self.turn_count,
                "context_used": context[:200] + "..." if len(context) > 200 else context
            }
            
        except Exception as e:
            logger.error(f"Error in conversation turn: {e}")
            return {
                "response": f"Hey! Sorry, I'm having a bit of a technical moment. Could you try that again? 😅 (Error: {str(e)[:50]})",
                "thread_id": self.thread_id,
                "extracted_facts": [],
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    async def _query_memory(self, user_input: str) -> str:
        """Query memory for relevant context (RAG simulation)."""
        # Simple keyword-based memory retrieval
        # In a full system, this would use vector embeddings
        
        relevant_facts = []
        relevant_messages = []
        
        # Find relevant facts
        user_lower = user_input.lower()
        for fact in self.extracted_facts[-20:]:  # Check recent facts
            fact_words = fact.lower().split()
            user_words = user_lower.split()
            
            # Simple word overlap check
            overlap = set(fact_words) & set(user_words)
            if len(overlap) > 1 or any(word in fact.lower() for word in ['name', 'work', 'live', 'like', 'love']):
                relevant_facts.append(fact)
        
        # Find relevant previous messages
        for msg in self.conversation_history[-10:]:  # Check recent messages
            if msg["role"] == "user":
                msg_words = msg["content"].lower().split()
                user_words = user_lower.split()
                overlap = set(msg_words) & set(user_words)
                if len(overlap) > 2:
                    relevant_messages.append(f"Previous: {msg['content'][:100]}")
        
        # Combine context
        context_parts = []
        if relevant_facts:
            context_parts.append("Relevant facts: " + "; ".join(relevant_facts[-5:]))
        if relevant_messages:
            context_parts.append("Related previous messages: " + "; ".join(relevant_messages[-3:]))
        
        return " | ".join(context_parts)
    
    async def _generate_response_with_context(self, user_input: str, context: str) -> str:
        """Generate Calum's response with full context."""
        if not self.llm_client:
            return "Hey there! I'm Calum, but I'm having some technical difficulties right now. My AI brain isn't fully connected! 🤖"
        
        # Build comprehensive prompt
        facts_context = ""
        if self.extracted_facts:
            facts_text = "\n".join([f"- {fact}" for fact in self.extracted_facts[-15:]])
            facts_context = f"\nFACTS I REMEMBER ABOUT YOU:\n{facts_text}\n"
        
        history_context = ""
        if self.compressed_history:
            history_context = f"\nCONVERSATION SUMMARY:\n{self.compressed_history}\n"
        
        recent_context = ""
        if len(self.conversation_history) > 2:
            recent = self.conversation_history[-6:]  # Last 3 exchanges
            recent_formatted = []
            for msg in recent:
                role = "You" if msg["role"] == "user" else "Calum"
                recent_formatted.append(f"{role}: {msg['content']}")
            recent_context = f"\nRECENT CONVERSATION:\n" + "\n".join(recent_formatted) + "\n"
        
        memory_context = ""
        if context:
            memory_context = f"\nRELEVANT CONTEXT FROM MEMORY:\n{context}\n"
        
        full_prompt = f"""{self.persona_prompt}

{facts_context}{history_context}{recent_context}{memory_context}
Current message from user: {user_input}

Respond as Calum, using all the context above to give a personalized, authentic response that shows you remember our relationship and previous conversations:"""

        try:
            if LANGCHAIN_AVAILABLE:
                response = await asyncio.to_thread(
                    self.llm_client.invoke,
                    [{"role": "user", "content": full_prompt}]
                )
                return response.content
            
            elif OPENAI_AVAILABLE:
                response = await asyncio.to_thread(
                    self.llm_client.chat.completions.create,
                    model=self.model_name,
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=self.temperature,
                    max_tokens=500
                )
                return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Hey! I'm having a bit of trouble thinking right now. Mind trying that again? 🤔"
    
    async def _extract_facts(self, user_input: str, response: str) -> List[str]:
        """Extract facts using LLM."""
        if not self.llm_client:
            return self._simple_fact_extraction(user_input)
        
        fact_prompt = f"""Extract any new factual information about the user from this conversation.
Return ONLY a JSON list of strings with the facts, or an empty list [] if none.
Focus on personal information, preferences, work, location, interests, etc.

User: {user_input}
Assistant: {response}

Facts (JSON list only):"""

        try:
            if LANGCHAIN_AVAILABLE:
                fact_response = await asyncio.to_thread(
                    self.llm_client.invoke,
                    [{"role": "user", "content": fact_prompt}]
                )
                content = fact_response.content
            elif OPENAI_AVAILABLE:
                fact_response = await asyncio.to_thread(
                    self.llm_client.chat.completions.create,
                    model=self.model_name,
                    messages=[{"role": "user", "content": fact_prompt}],
                    temperature=0.1,
                    max_tokens=200
                )
                content = fact_response.choices[0].message.content
            
            # Parse JSON response
            try:
                facts = json.loads(content.strip())
                if isinstance(facts, list):
                    return [fact for fact in facts if isinstance(fact, str) and len(fact.strip()) > 5]
            except json.JSONDecodeError:
                # Fallback to simple extraction
                return self._simple_fact_extraction(user_input)
                
        except Exception as e:
            logger.warning(f"Fact extraction failed: {e}")
            return self._simple_fact_extraction(user_input)
        
        return []
    
    def _simple_fact_extraction(self, user_input: str) -> List[str]:
        """Simple keyword-based fact extraction as fallback."""
        new_facts = []
        
        fact_indicators = [
            "my name is", "i'm", "i am", "i work", "i live", "i love", 
            "i like", "i hate", "i have", "my", "i study", "i'm from"
        ]
        
        user_lower = user_input.lower()
        for indicator in fact_indicators:
            if indicator in user_lower:
                sentences = user_input.split('.')
                for sentence in sentences:
                    if indicator in sentence.lower():
                        fact = sentence.strip()
                        if fact and len(fact) > 10:
                            new_facts.append(fact)
                        break
        
        return new_facts[-2:]  # Return max 2 new facts
    
    async def _summarize_conversation(self):
        """Summarize conversation when it gets too long."""
        if not self.llm_client or len(self.conversation_history) < 8:
            return
        
        # Get messages to summarize (exclude the most recent ones)
        messages_to_summarize = self.conversation_history[:-self.turns_to_keep_after_summary]
        
        if not messages_to_summarize:
            return
        
        # Format conversation for summarization
        conversation_text = ""
        for msg in messages_to_summarize:
            role = "User" if msg["role"] == "user" else "Calum"
            conversation_text += f"{role}: {msg['content']}\n"
        
        summary_prompt = f"""Summarize this conversation between Calum and the user, focusing on:
1. Key facts learned about the user
2. Important topics discussed
3. The relationship/rapport built
4. Any ongoing conversations or plans

Previous summary: {self.compressed_history if self.compressed_history else 'None'}

Conversation to summarize:
{conversation_text}

Comprehensive summary:"""

        try:
            if LANGCHAIN_AVAILABLE:
                summary_response = await asyncio.to_thread(
                    self.llm_client.invoke,
                    [{"role": "user", "content": summary_prompt}]
                )
                new_summary = summary_response.content
            elif OPENAI_AVAILABLE:
                summary_response = await asyncio.to_thread(
                    self.llm_client.chat.completions.create,
                    model=self.model_name,
                    messages=[{"role": "user", "content": summary_prompt}],
                    temperature=0.1,
                    max_tokens=300
                )
                new_summary = summary_response.choices[0].message.content
            
            # Update compressed history
            self.compressed_history = new_summary
            self._save_compressed_history(new_summary)
            
            # Keep only recent messages in memory
            self.conversation_history = self.conversation_history[-self.turns_to_keep_after_summary:]
            
            logger.info(f"Conversation summarized. Keeping {len(self.conversation_history)} recent messages.")
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
    
    async def _cleanup(self):
        """Cleanup method."""
        pass

# Global conversation systems cache for FastAPI
conversation_systems: Dict[str, CompleteCalumSystem] = {}

def get_complete_conversation_system(thread_id: str, session: Dict) -> CompleteCalumSystem:
    """Get or create a complete conversation system."""
    if thread_id not in conversation_systems:
        session["thread_id"] = thread_id
        conversation_systems[thread_id] = CompleteCalumSystem(session)
    
    return conversation_systems[thread_id]
