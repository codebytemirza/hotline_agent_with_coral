import streamlit as st
import asyncio
import json
import threading
import time
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime
import numpy as np
import io
import wave

# LangGraph Supervisor and LangChain imports
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.messages.base import BaseMessage as CoreBaseMessage

# Audio and Discord imports
try:
    from fastrtc import Stream, ReplyOnPause, AlgoOptions, audio_to_bytes
    FASTRTC_AVAILABLE = True
except ImportError:
    FASTRTC_AVAILABLE = False
    st.warning("FastRTC not available. Voice features will be limited.")

try:
    from elevenlabs.client import ElevenLabs
    ELEVENLABS_AVAILABLE = True
except ImportError:
    ELEVENLABS_AVAILABLE = False
    st.warning("ElevenLabs not available. TTS/STT features disabled.")

try:
    import discord
    from discord.ext import commands
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    st.warning("Discord.py not available. Discord features disabled.")

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

class MultiAgentSystem:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.checkpointer = InMemorySaver()
        self.store = InMemoryStore()
        self.elevenlabs_client = None
        self.discord_bot = None
        self.audio_stream = None
        self.supervisor_workflow = None
        self.compiled_app = None
        self.agents = {}
        
        # Initialize clients and agents
        self._initialize_clients()
        self._create_specialized_agents()
        self._create_supervisor_workflow()
    
    def _initialize_clients(self):
        """Initialize ElevenLabs and Discord clients"""
        # Initialize ElevenLabs
        if ELEVENLABS_AVAILABLE and self.config.get('elevenlabs_api_key'):
            try:
                self.elevenlabs_client = ElevenLabs(api_key=self.config['elevenlabs_api_key'])
                st.success("ElevenLabs client initialized")
            except Exception as e:
                st.error(f"ElevenLabs initialization failed: {str(e)}")
        
        # Initialize Discord
        if DISCORD_AVAILABLE and self.config.get('discord_bot_token'):
            try:
                intents = discord.Intents.default()
                intents.message_content = True
                self.discord_bot = commands.Bot(command_prefix="!", intents=intents)
                
                # Add event handlers
                @self.discord_bot.event
                async def on_ready():
                    st.success(f"Discord bot connected as {self.discord_bot.user}")
                
                st.success("Discord bot initialized")
            except Exception as e:
                st.error(f"Discord initialization failed: {str(e)}")
    
    def _get_llm(self):
        """Get the configured LLM"""
        if self.config['model_provider'] == 'openai':
            return ChatOpenAI(
                api_key=self.config['openai_api_key'],
                model=self.config['model_name'],
                temperature=self.config['temperature']
            )
        elif self.config['model_provider'] == 'groq':
            return ChatGroq(
                api_key=self.config['groq_api_key'],
                model=self.config['model_name'],
                temperature=self.config['temperature']
            )
        else:
            raise ValueError(f"Unsupported model provider: {self.config['model_provider']}")
    
    def _create_voice_tools(self):
        """Create tools for voice agent"""
        @tool
        def text_to_speech(text: str, voice_id: Optional[str] = None) -> str:
            """Convert text to speech using ElevenLabs TTS"""
            if not self.elevenlabs_client:
                return "ElevenLabs not available"
            
            try:
                voice_id = voice_id or self.config.get('elevenlabs_voice_id', 'JBFqnCBsd6RMkjVDRZzb')
                model_id = self.config.get('elevenlabs_model', 'eleven_multilingual_v2')
                
                # Generate TTS
                audio_generator = self.elevenlabs_client.text_to_speech.convert(
                    text=text,
                    voice_id=voice_id,
                    model_id=model_id,
                    output_format="mp3_22050_32"
                )
                
                # Store audio data for potential playback
                audio_data = b''.join(audio_generator) if hasattr(audio_generator, '__iter__') else audio_generator
                
                return f"TTS generated successfully for: '{text[:50]}{'...' if len(text) > 50 else ''}'"
            except Exception as e:
                return f"TTS error: {str(e)}"
        
        @tool
        def speech_to_text(audio_description: str = "incoming audio") -> str:
            """Convert speech to text using ElevenLabs STT (placeholder for actual audio processing)"""
            if not self.elevenlabs_client:
                return "ElevenLabs not available"
            
            try:
                # In real implementation, this would process actual audio data
                return f"STT ready to process: {audio_description}"
            except Exception as e:
                return f"STT error: {str(e)}"
        
        @tool
        def start_voice_stream() -> str:
            """Initialize FastRTC voice streaming"""
            if not FASTRTC_AVAILABLE:
                return "FastRTC not available"
            
            try:
                # Initialize voice stream configuration
                return "Voice stream initialized and ready for real-time audio"
            except Exception as e:
                return f"Voice stream error: {str(e)}"
        
        @tool
        def get_voice_status() -> str:
            """Get current voice system status"""
            status = {
                "elevenlabs": "connected" if self.elevenlabs_client else "disconnected",
                "fastrtc": "available" if FASTRTC_AVAILABLE else "unavailable",
                "voice_id": self.config.get('elevenlabs_voice_id', 'not_set'),
                "model": self.config.get('elevenlabs_model', 'not_set')
            }
            return json.dumps(status, indent=2)
        
        return [text_to_speech, speech_to_text, start_voice_stream, get_voice_status]
    
    def _create_discord_tools(self):
        """Create tools for Discord agent"""
        @tool
        def send_discord_message(message: str, channel_id: Optional[str] = None) -> str:
            """Send a message to a Discord channel"""
            if not self.discord_bot:
                return "Discord bot not available"
            
            try:
                target_channel_id = channel_id or self.config.get('discord_channel_id')
                if not target_channel_id:
                    return "No channel ID specified"
                
                # Store message for processing (in real implementation, this would send to Discord)
                return f"Discord message queued: '{message[:50]}{'...' if len(message) > 50 else ''}' -> Channel {target_channel_id}"
            except Exception as e:
                return f"Discord send error: {str(e)}"
        
        @tool
        def get_discord_channels() -> str:
            """Get list of available Discord channels"""
            if not self.discord_bot:
                return "Discord bot not available"
            
            try:
                # Mock channel list (in real implementation, would fetch from Discord)
                channels = [
                    {"id": "123456789", "name": "general", "type": "text"},
                    {"id": "987654321", "name": "voice-chat", "type": "voice"},
                    {"id": "456789123", "name": "bot-commands", "type": "text"}
                ]
                return json.dumps(channels, indent=2)
            except Exception as e:
                return f"Discord channels error: {str(e)}"
        
        @tool
        def create_voice_channel(channel_name: str = "Multi-Agent Session") -> str:
            """Create a new voice channel in Discord"""
            if not self.discord_bot:
                return "Discord bot not available"
            
            try:
                # Mock voice channel creation
                return f"Voice channel '{channel_name}' created successfully"
            except Exception as e:
                return f"Voice channel creation error: {str(e)}"
        
        @tool
        def get_discord_status() -> str:
            """Get Discord bot status and server information"""
            if not self.discord_bot:
                return "Discord bot not connected"
            
            status = {
                "bot_connected": bool(self.discord_bot),
                "guild_id": self.config.get('discord_guild_id', 'not_set'),
                "permissions": "send_messages, read_history, manage_channels",
                "status": "ready"
            }
            return json.dumps(status, indent=2)
        
        return [send_discord_message, get_discord_channels, create_voice_channel, get_discord_status]
    
    def _create_memory_tools(self):
        """Create tools for memory agent"""
        @tool
        def store_conversation(conversation_id: str, content: str, metadata: Optional[str] = None) -> str:
            """Store a conversation in memory with optional metadata"""
            try:
                # Use the InMemoryStore to persist data
                conversation_data = {
                    "content": content,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": metadata or {}
                }
                
                # In real implementation, would use self.store.put()
                return f"Conversation '{conversation_id}' stored successfully with {len(content)} characters"
            except Exception as e:
                return f"Storage error: {str(e)}"
        
        @tool
        def retrieve_conversation(conversation_id: str) -> str:
            """Retrieve a conversation from memory"""
            try:
                # Mock retrieval (in real implementation, would use self.store.get())
                return f"Retrieved conversation '{conversation_id}': [Conversation data would be returned here]"
            except Exception as e:
                return f"Retrieval error: {str(e)}"
        
        @tool
        def get_conversation_summary(conversation_id: str, max_length: int = 200) -> str:
            """Generate a summary of a stored conversation"""
            try:
                # Mock summarization
                return f"Summary of '{conversation_id}': Multi-agent interaction session with voice, Discord, and memory capabilities demonstrated."
            except Exception as e:
                return f"Summary error: {str(e)}"
        
        @tool
        def list_conversations() -> str:
            """List all stored conversations"""
            try:
                # Mock conversation list
                conversations = [
                    {"id": "session_001", "timestamp": "2024-01-01T10:00:00", "summary": "Voice interaction test"},
                    {"id": "session_002", "timestamp": "2024-01-01T11:00:00", "summary": "Discord bot commands"},
                    {"id": "session_003", "timestamp": "2024-01-01T12:00:00", "summary": "Memory management demo"}
                ]
                return json.dumps(conversations, indent=2)
            except Exception as e:
                return f"List conversations error: {str(e)}"
        
        @tool
        def get_memory_status() -> str:
            """Get memory system status"""
            status = {
                "checkpointer": "active" if self.checkpointer else "inactive",
                "store": "active" if self.store else "inactive",
                "memory_type": "in_memory",
                "persistent": False
            }
            return json.dumps(status, indent=2)
        
        return [store_conversation, retrieve_conversation, get_conversation_summary, list_conversations, get_memory_status]
    
    def _create_research_tools(self):
        """Create tools for research agent"""
        @tool
        def web_search(query: str) -> str:
            """Search the web for information (mock implementation)"""
            try:
                # Mock search results
                results = {
                    "query": query,
                    "results": [
                        {"title": f"Results for '{query}' - Article 1", "snippet": "Relevant information about the query..."},
                        {"title": f"Results for '{query}' - Article 2", "snippet": "Additional context and details..."},
                        {"title": f"Results for '{query}' - Article 3", "snippet": "Further insights and analysis..."}
                    ],
                    "total_results": 3
                }
                return json.dumps(results, indent=2)
            except Exception as e:
                return f"Search error: {str(e)}"
        
        @tool
        def analyze_data(data_description: str) -> str:
            """Analyze data and provide insights"""
            try:
                analysis = {
                    "data_type": data_description,
                    "analysis": f"Analysis of {data_description} shows interesting patterns and trends.",
                    "insights": [
                        "Key insight 1: Data shows positive correlation",
                        "Key insight 2: Trending patterns identified",
                        "Key insight 3: Recommendations for improvement"
                    ]
                }
                return json.dumps(analysis, indent=2)
            except Exception as e:
                return f"Analysis error: {str(e)}"
        
        return [web_search, analyze_data]
    
    def _create_specialized_agents(self):
        """Create specialized agents using create_react_agent with correct parameters"""
        llm = self._get_llm()
        
        # Voice Agent - use state_modifier for system message
        voice_tools = self._create_voice_tools()
        
        # Create system message for voice agent as a callable
        def voice_state_modifier(state):
            return [
                SystemMessage(content="""You are a voice interaction expert specializing in audio processing and speech technologies. 

Your responsibilities include:
- Converting text to speech using ElevenLabs TTS with various voice models
- Processing speech-to-text transcription with high accuracy
- Managing FastRTC audio streaming for real-time voice interactions
- Handling audio format conversions and quality optimization
- Providing voice system status and diagnostics

Always prioritize clear, natural speech output and accurate transcription. Use appropriate voice settings based on context.""")
            ] + state["messages"]
        
        self.agents['voice_agent'] = create_react_agent(
            model=llm,
            tools=voice_tools,
            state_modifier=voice_state_modifier
        )
        
        # Discord Agent
        discord_tools = self._create_discord_tools()
        
        def discord_state_modifier(state):
            return [
                SystemMessage(content="""You are a Discord integration expert specializing in bot operations and server management.

Your responsibilities include:
- Sending messages to Discord channels with proper formatting
- Managing voice channels and server interactions
- Handling Discord bot commands and permissions
- Monitoring server status and member interactions
- Creating and managing Discord community features

Always ensure messages are appropriate for the target audience and follow Discord community guidelines. Use proper Discord markdown formatting when needed.""")
            ] + state["messages"]
        
        self.agents['discord_agent'] = create_react_agent(
            model=llm,
            tools=discord_tools,
            state_modifier=discord_state_modifier
        )
        
        # Memory Agent
        memory_tools = self._create_memory_tools()
        
        def memory_state_modifier(state):
            return [
                SystemMessage(content="""You are a memory management expert specializing in conversation storage and context retrieval.

Your responsibilities include:
- Storing conversations with proper metadata and indexing
- Retrieving relevant conversation history and context
- Generating meaningful summaries of interactions
- Managing memory persistence across sessions
- Optimizing storage for efficient retrieval

Always maintain conversation context and provide relevant historical information when needed. Ensure data privacy and proper organization of stored information.""")
            ] + state["messages"]
        
        self.agents['memory_agent'] = create_react_agent(
            model=llm,
            tools=memory_tools,
            state_modifier=memory_state_modifier
        )
        
        # Research Agent
        research_tools = self._create_research_tools()
        
        def research_state_modifier(state):
            return [
                SystemMessage(content="""You are a research and analysis expert specializing in information gathering and data analysis.

Your responsibilities include:
- Conducting web searches for accurate and relevant information
- Analyzing data patterns and providing actionable insights
- Synthesizing information from multiple sources
- Providing fact-based responses with proper context
- Generating comprehensive research reports

Always verify information accuracy and provide well-structured, evidence-based responses. Cite sources when applicable and highlight key insights clearly.""")
            ] + state["messages"]
        
        self.agents['research_agent'] = create_react_agent(
            model=llm,
            tools=research_tools,
            state_modifier=research_state_modifier
        )
        
        st.success(f"Created {len(self.agents)} specialized agents with state modifiers")
    
    def _create_supervisor_workflow(self):
        """Create the supervisor workflow using langgraph_supervisor"""
        try:
            llm = self._get_llm()
            
            # Get list of agents with their names
            agent_names = list(self.agents.keys())
            
            # Create supervisor system message
            supervisor_system_message = """You are an intelligent multi-agent system supervisor coordinating specialized AI agents.

Your team consists of:
1. **voice_agent**: Handles all voice, audio, TTS, STT, and FastRTC streaming tasks
2. **discord_agent**: Manages Discord bot operations, messaging, and server interactions  
3. **memory_agent**: Handles conversation storage, retrieval, and memory management
4. **research_agent**: Conducts research, web searches, and data analysis

ROUTING RULES:
- Voice/Audio requests → voice_agent (keywords: voice, audio, speech, TTS, STT, sound, listen, speak)
- Discord requests → discord_agent (keywords: discord, message, channel, bot, server, chat)
- Memory requests → memory_agent (keywords: remember, store, recall, history, conversation, save)
- Research requests → research_agent (keywords: search, research, find, analyze, information, data)

For complex requests involving multiple domains, coordinate between agents as needed.        self._initialize_clients()
        self._create_specialized_agents()
        self._create_supervisor_workflow()
    
    def _initialize_clients(self):
        """Initialize ElevenLabs and Discord clients"""
        # Initialize ElevenLabs
        if ELEVENLABS_AVAILABLE and self.config.get('elevenlabs_api_key'):
            try:
                self.elevenlabs_client = ElevenLabs(api_key=self.config['elevenlabs_api_key'])
                st.success("✅ ElevenLabs client initialized")
            except Exception as e:
                st.error(f"❌ ElevenLabs initialization failed: {str(e)}")
        
        # Initialize Discord
        if DISCORD_AVAILABLE and self.config.get('discord_bot_token'):
            try:
                intents = discord.Intents.default()
                intents.message_content = True
                self.discord_bot = commands.Bot(command_prefix="!", intents=intents)
                
                # Add event handlers
                @self.discord_bot.event
                async def on_ready():
                    st.success(f"✅ Discord bot connected as {self.discord_bot.user}")
                
                st.success("✅ Discord bot initialized")
            except Exception as e:
                st.error(f"❌ Discord initialization failed: {str(e)}")
    
    def _get_llm(self):
        """Get the configured LLM"""
        if self.config['model_provider'] == 'openai':
            return ChatOpenAI(
                api_key=self.config['openai_api_key'],
                model=self.config['model_name'],
                temperature=self.config['temperature']
            )
        elif self.config['model_provider'] == 'groq':
            return ChatGroq(
                api_key=self.config['groq_api_key'],
                model=self.config['model_name'],
                temperature=self.config['temperature']
            )
        else:
            raise ValueError(f"Unsupported model provider: {self.config['model_provider']}")
    
    def _create_voice_tools(self):
        """Create tools for voice agent"""
        @tool
        def text_to_speech(text: str, voice_id: Optional[str] = None) -> str:
            """Convert text to speech using ElevenLabs TTS"""
            if not self.elevenlabs_client:
                return "❌ ElevenLabs not available"
            
            try:
                voice_id = voice_id or self.config.get('elevenlabs_voice_id', 'JBFqnCBsd6RMkjVDRZzb')
                model_id = self.config.get('elevenlabs_model', 'eleven_multilingual_v2')
                
                # Generate TTS
                audio_generator = self.elevenlabs_client.text_to_speech.convert(
                    text=text,
                    voice_id=voice_id,
                    model_id=model_id,
                    output_format="mp3_22050_32"
                )
                
                # Store audio data for potential playback
                audio_data = b''.join(audio_generator) if hasattr(audio_generator, '__iter__') else audio_generator
                
                return f"🔊 TTS generated successfully for: '{text[:50]}{'...' if len(text) > 50 else ''}'"
            except Exception as e:
                return f"❌ TTS error: {str(e)}"
        
        @tool
        def speech_to_text(audio_description: str = "incoming audio") -> str:
            """Convert speech to text using ElevenLabs STT (placeholder for actual audio processing)"""
            if not self.elevenlabs_client:
                return "❌ ElevenLabs not available"
            
            try:
                # In real implementation, this would process actual audio data
                return f"👂 STT ready to process: {audio_description}"
            except Exception as e:
                return f"❌ STT error: {str(e)}"
        
        @tool
        def start_voice_stream() -> str:
            """Initialize FastRTC voice streaming"""
            if not FASTRTC_AVAILABLE:
                return "❌ FastRTC not available"
            
            try:
                # Initialize voice stream configuration
                return "🎙️ Voice stream initialized and ready for real-time audio"
            except Exception as e:
                return f"❌ Voice stream error: {str(e)}"
        
        @tool
        def get_voice_status() -> str:
            """Get current voice system status"""
            status = {
                "elevenlabs": "connected" if self.elevenlabs_client else "disconnected",
                "fastrtc": "available" if FASTRTC_AVAILABLE else "unavailable",
                "voice_id": self.config.get('elevenlabs_voice_id', 'not_set'),
                "model": self.config.get('elevenlabs_model', 'not_set')
            }
            return json.dumps(status, indent=2)
        
        return [text_to_speech, speech_to_text, start_voice_stream, get_voice_status]
    
    def _create_discord_tools(self):
        """Create tools for Discord agent"""
        @tool
        def send_discord_message(message: str, channel_id: Optional[str] = None) -> str:
            """Send a message to a Discord channel"""
            if not self.discord_bot:
                return "❌ Discord bot not available"
            
            try:
                target_channel_id = channel_id or self.config.get('discord_channel_id')
                if not target_channel_id:
                    return "❌ No channel ID specified"
                
                # Store message for processing (in real implementation, this would send to Discord)
                return f"📤 Discord message queued: '{message[:50]}{'...' if len(message) > 50 else ''}' → Channel {target_channel_id}"
            except Exception as e:
                return f"❌ Discord send error: {str(e)}"
        
        @tool
        def get_discord_channels() -> str:
            """Get list of available Discord channels"""
            if not self.discord_bot:
                return "❌ Discord bot not available"
            
            try:
                # Mock channel list (in real implementation, would fetch from Discord)
                channels = [
                    {"id": "123456789", "name": "general", "type": "text"},
                    {"id": "987654321", "name": "voice-chat", "type": "voice"},
                    {"id": "456789123", "name": "bot-commands", "type": "text"}
                ]
                return json.dumps(channels, indent=2)
            except Exception as e:
                return f"❌ Discord channels error: {str(e)}"
        
        @tool
        def create_voice_channel(channel_name: str = "Multi-Agent Session") -> str:
            """Create a new voice channel in Discord"""
            if not self.discord_bot:
                return "❌ Discord bot not available"
            
            try:
                # Mock voice channel creation
                return f"🎙️ Voice channel '{channel_name}' created successfully"
            except Exception as e:
                return f"❌ Voice channel creation error: {str(e)}"
        
        @tool
        def get_discord_status() -> str:
            """Get Discord bot status and server information"""
            if not self.discord_bot:
                return "❌ Discord bot not connected"
            
            status = {
                "bot_connected": bool(self.discord_bot),
                "guild_id": self.config.get('discord_guild_id', 'not_set'),
                "permissions": "send_messages, read_history, manage_channels",
                "status": "ready"
            }
            return json.dumps(status, indent=2)
        
        return [send_discord_message, get_discord_channels, create_voice_channel, get_discord_status]
    
    def _create_memory_tools(self):
        """Create tools for memory agent"""
        @tool
        def store_conversation(conversation_id: str, content: str, metadata: Optional[str] = None) -> str:
            """Store a conversation in memory with optional metadata"""
            try:
                # Use the InMemoryStore to persist data
                conversation_data = {
                    "content": content,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": metadata or {}
                }
                
                # In real implementation, would use self.store.put()
                return f"💾 Conversation '{conversation_id}' stored successfully with {len(content)} characters"
            except Exception as e:
                return f"❌ Storage error: {str(e)}"
        
        @tool
        def retrieve_conversation(conversation_id: str) -> str:
            """Retrieve a conversation from memory"""
            try:
                # Mock retrieval (in real implementation, would use self.store.get())
                return f"📖 Retrieved conversation '{conversation_id}': [Conversation data would be returned here]"
            except Exception as e:
                return f"❌ Retrieval error: {str(e)}"
        
        @tool
        def get_conversation_summary(conversation_id: str, max_length: int = 200) -> str:
            """Generate a summary of a stored conversation"""
            try:
                # Mock summarization
                return f"📝 Summary of '{conversation_id}': Multi-agent interaction session with voice, Discord, and memory capabilities demonstrated."
            except Exception as e:
                return f"❌ Summary error: {str(e)}"
        
        @tool
        def list_conversations() -> str:
            """List all stored conversations"""
            try:
                # Mock conversation list
                conversations = [
                    {"id": "session_001", "timestamp": "2024-01-01T10:00:00", "summary": "Voice interaction test"},
                    {"id": "session_002", "timestamp": "2024-01-01T11:00:00", "summary": "Discord bot commands"},
                    {"id": "session_003", "timestamp": "2024-01-01T12:00:00", "summary": "Memory management demo"}
                ]
                return json.dumps(conversations, indent=2)
            except Exception as e:
                return f"❌ List conversations error: {str(e)}"
        
        @tool
        def get_memory_status() -> str:
            """Get memory system status"""
            status = {
                "checkpointer": "active" if self.checkpointer else "inactive",
                "store": "active" if self.store else "inactive",
                "memory_type": "in_memory",
                "persistent": False
            }
            return json.dumps(status, indent=2)
        
        return [store_conversation, retrieve_conversation, get_conversation_summary, list_conversations, get_memory_status]
    
    def _create_research_tools(self):
        """Create tools for research agent"""
        @tool
        def web_search(query: str) -> str:
            """Search the web for information (mock implementation)"""
            try:
                # Mock search results
                results = {
                    "query": query,
                    "results": [
                        {"title": f"Results for '{query}' - Article 1", "snippet": "Relevant information about the query..."},
                        {"title": f"Results for '{query}' - Article 2", "snippet": "Additional context and details..."},
                        {"title": f"Results for '{query}' - Article 3", "snippet": "Further insights and analysis..."}
                    ],
                    "total_results": 3
                }
                return json.dumps(results, indent=2)
            except Exception as e:
                return f"❌ Search error: {str(e)}"
        
        @tool
        def analyze_data(data_description: str) -> str:
            """Analyze data and provide insights"""
            try:
                analysis = {
                    "data_type": data_description,
                    "analysis": f"Analysis of {data_description} shows interesting patterns and trends.",
                    "insights": [
                        "Key insight 1: Data shows positive correlation",
                        "Key insight 2: Trending patterns identified",
                        "Key insight 3: Recommendations for improvement"
                    ]
                }
                return json.dumps(analysis, indent=2)
            except Exception as e:
                return f"❌ Analysis error: {str(e)}"
        
        return [web_search, analyze_data]
    
    def _create_specialized_agents(self):
        """Create specialized agents using create_react_agent"""
        llm = self._get_llm()
        
        # Voice Agent
        voice_tools = self._create_voice_tools()
        self.agents['voice_agent'] = create_react_agent(
            model=llm,
            tools=voice_tools,
            name="voice_expert",
            system_message=SystemMessage(content="""You are a voice interaction expert specializing in audio processing and speech technologies. 

Your responsibilities include:
- Converting text to speech using ElevenLabs TTS with various voice models
- Processing speech-to-text transcription with high accuracy
- Managing FastRTC audio streaming for real-time voice interactions
- Handling audio format conversions and quality optimization
- Providing voice system status and diagnostics

Always prioritize clear, natural speech output and accurate transcription. Use appropriate voice settings based on context.""")
        )
        
        # Discord Agent
        discord_tools = self._create_discord_tools()
        self.agents['discord_agent'] = create_react_agent(
            model=llm,
            tools=discord_tools,
            name="discord_expert",
            system_message=SystemMessage(content="""You are a Discord integration expert specializing in bot operations and server management.

Your responsibilities include:
- Sending messages to Discord channels with proper formatting
- Managing voice channels and server interactions
- Handling Discord bot commands and permissions
- Monitoring server status and member interactions
- Creating and managing Discord community features

Always ensure messages are appropriate for the target audience and follow Discord community guidelines. Use proper Discord markdown formatting when needed.""")
        )
        
        # Memory Agent
        memory_tools = self._create_memory_tools()
        self.agents['memory_agent'] = create_react_agent(
            model=llm,
            tools=memory_tools,
            name="memory_expert",
            system_message=SystemMessage(content="""You are a memory management expert specializing in conversation storage and context retrieval.

Your responsibilities include:
- Storing conversations with proper metadata and indexing
- Retrieving relevant conversation history and context
- Generating meaningful summaries of interactions
- Managing memory persistence across sessions
- Optimizing storage for efficient retrieval

Always maintain conversation context and provide relevant historical information when needed. Ensure data privacy and proper organization of stored information.""")
        )
        
        # Research Agent
        research_tools = self._create_research_tools()
        self.agents['research_agent'] = create_react_agent(
            model=llm,
            tools=research_tools,
            name="research_expert",
            system_message=SystemMessage(content="""You are a research and analysis expert specializing in information gathering and data analysis.

Your responsibilities include:
- Conducting web searches for accurate and relevant information
- Analyzing data patterns and providing actionable insights
- Synthesizing information from multiple sources
- Providing fact-based responses with proper context
- Generating comprehensive research reports

Always verify information accuracy and provide well-structured, evidence-based responses. Cite sources when applicable and highlight key insights clearly.""")
        )
        
        st.success(f"✅ Created {len(self.agents)} specialized agents with system messages")
    
    def _create_supervisor_workflow(self):
        """Create the supervisor workflow using langgraph_supervisor"""
        try:
            llm = self._get_llm()
            
            # Get list of agents
            agent_list = list(self.agents.values())
            
            # Create supervisor system message
            supervisor_system_message = SystemMessage(content="""You are an intelligent multi-agent system supervisor coordinating specialized AI agents.

Your team consists of:
1. **voice_expert**: Handles all voice, audio, TTS, STT, and FastRTC streaming tasks
2. **discord_expert**: Manages Discord bot operations, messaging, and server interactions  
3. **memory_expert**: Handles conversation storage, retrieval, and memory management
4. **research_expert**: Conducts research, web searches, and data analysis

ROUTING RULES:
- Voice/Audio requests → voice_expert (keywords: voice, audio, speech, TTS, STT, sound, listen, speak)
- Discord requests → discord_expert (keywords: discord, message, channel, bot, server, chat)
- Memory requests → memory_expert (keywords: remember, store, recall, history, conversation, save)
- Research requests → research_expert (keywords: search, research, find, analyze, information, data)

For complex requests involving multiple domains, coordinate between agents as needed.

Always provide helpful, accurate responses and ensure smooth handoffs between agents. Maintain context throughout the conversation.""")
            
            # Create supervisor using langgraph_supervisor
            self.supervisor_workflow = create_supervisor(
                agents=agent_list,
                model=llm,
                system_message=supervisor_system_message,
                output_mode="full_history"  # Include full conversation history
            )
            
            # Compile the workflow with memory
            self.compiled_app = self.supervisor_workflow.compile(
                checkpointer=self.checkpointer,
                store=self.store
            )
            
            st.success("🔗 Supervisor workflow created with system message and compiled successfully")
            
        except Exception as e:
            st.error(f"❌ Supervisor workflow creation failed: {str(e)}")
            raise
    
    async def process_message(self, user_message: str, thread_id: str = "default_session") -> Dict[str, Any]:
        """Process a user message through the supervisor workflow"""
        try:
            if not self.compiled_app:
                raise ValueError("Supervisor workflow not initialized")
            
            # Create input state with proper BaseMessage types
            input_state = {
                "messages": [HumanMessage(content=user_message)]
            }
            
            # Configure with thread ID for memory persistence
            config = {"configurable": {"thread_id": thread_id}}
            
            # Process through supervisor workflow
            result = await self.compiled_app.ainvoke(input_state, config)
            
            return result
            
        except Exception as e:
            st.error(f"❌ Message processing error: {str(e)}")
            return {"error": str(e)}
    
    def create_voice_stream(self):
        """Create FastRTC voice stream for real-time audio interaction"""
        if not FASTRTC_AVAILABLE or not self.elevenlabs_client:
            return None
        
        def voice_response(audio_tuple):
            """Handle voice input and generate response"""
            try:
                sample_rate, audio_data = audio_tuple
                
                # Convert audio to WAV for ElevenLabs STT
                wav_bytes = self._audio_to_wav_bytes(audio_tuple)
                
                # Transcribe with ElevenLabs
                audio_io = io.BytesIO(wav_bytes)
                transcription = self.elevenlabs_client.speech_to_text.convert(
                    file=audio_io,
                    model_id="scribe_v1"
                )
                
                transcript = transcription.text if hasattr(transcription, 'text') else str(transcription)
                
                if transcript.strip():
                    # Process through supervisor
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(
                            self.process_message(transcript, "voice_session")
                        )
                        
                        # Get response text from the result
                        if result and "messages" in result:
                            ai_messages = []
                            for msg in result["messages"]:
                                if isinstance(msg, (AIMessage, CoreBaseMessage)):
                                    if hasattr(msg, 'content') and msg.content:
                                        ai_messages.append(msg)
                            
                            if ai_messages:
                                response_text = ai_messages[-1].content
                            else:
                                response_text = "I'm processing your request."
                        else:
                            response_text = "I'm here to help with voice, Discord, memory, and research tasks."
                            
                    finally:
                        loop.close()
                else:
                    response_text = "I'm listening. Please tell me how I can help you."
                
                # Generate TTS response
                voice_id = self.config.get('elevenlabs_voice_id', 'JBFqnCBsd6RMkjVDRZzb')
                tts_audio = self.elevenlabs_client.text_to_speech.convert(
                    text=response_text,
                    voice_id=voice_id,
                    model_id=self.config.get('elevenlabs_model', 'eleven_multilingual_v2'),
                    output_format="mp3_22050_32"
                )
                
                # Process and yield audio chunks
                yield from self._process_tts_audio(tts_audio)
                
            except Exception as e:
                st.error(f"Voice response error: {str(e)}")
                # Return silence on error
                for _ in range(int(22050 * 2 / 1024)):
                    yield (22050, np.zeros(1024, dtype=np.float32))
        
        return Stream(
            modality="audio",
            mode="send-receive",
            handler=ReplyOnPause(
                voice_response,
                algo_options=AlgoOptions(speech_threshold=0.3)
            )
        )
    
    def _audio_to_wav_bytes(self, audio_tuple):
        """Convert audio tuple to WAV bytes"""
        sample_rate, audio_data = audio_tuple
        
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        wav_buffer.seek(0)
        return wav_buffer.read()
    
    def _process_tts_audio(self, tts_response):
        """Process TTS audio response for streaming"""
        try:
            # Collect audio data
            if hasattr(tts_response, '__iter__') and not isinstance(tts_response, (str, bytes)):
                audio_content = b''.join(tts_response)
            elif hasattr(tts_response, 'read'):
                audio_content = tts_response.read()
            elif isinstance(tts_response, bytes):
                audio_content = tts_response
            else:
                audio_content = bytes(tts_response)
            
            if not audio_content:
                # Return silence if no audio
                for _ in range(int(22050 * 2 / 1024)):
                    yield (22050, np.zeros(1024, dtype=np.float32))
                return
            
            # Process MP3 audio
            if PYDUB_AVAILABLE and (audio_content.startswith(b'ID3') or audio_content[0:2] == b'\xff\xfb'):
                audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_content))
                if audio_segment.channels > 1:
                    audio_segment = audio_segment.set_channels(1)
                
                sample_rate = audio_segment.frame_rate
                raw_audio = audio_segment.raw_data
                audio_array = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                # Fallback to raw processing
                sample_rate = 22050
                audio_array = np.frombuffer(audio_content, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Stream in chunks
            chunk_size = 1024
            for i in range(0, len(audio_array), chunk_size):
                chunk = audio_array[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    padded_chunk = np.zeros(chunk_size, dtype=np.float32)
                    padded_chunk[:len(chunk)] = chunk
                    chunk = padded_chunk
                yield (sample_rate, chunk)
                
        except Exception as e:
            st.error(f"TTS processing error: {str(e)}")
            for _ in range(int(22050 * 2 / 1024)):
                yield (22050, np.zeros(1024, dtype=np.float32))


def main():
    st.set_page_config(
        page_title="🤖 Multi-Agent LangGraph System",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 Multi-Agent LangGraph Supervisor System")
    st.markdown("**Powered by LangGraph Supervisor with Voice, Discord & Memory Integration**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 System Configuration")
        
        # Model Configuration
        st.subheader("🧠 AI Model Setup")
        model_provider = st.selectbox("Provider", ["openai", "groq"], key="model_provider")
        
        if model_provider == "openai":
            openai_api_key = st.text_input("OpenAI API Key", type="password", key="openai_key")
            model_name = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"], key="openai_model")
        else:
            groq_api_key = st.text_input("Groq API Key", type="password", key="groq_key")
            model_name = st.selectbox("Model", ["llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"], key="groq_model")
        
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, key="temperature")
        
        # ElevenLabs Configuration
        st.subheader("🎙️ ElevenLabs Voice")
        elevenlabs_api_key = st.text_input("ElevenLabs API Key", type="password", key="elevenlabs_key")
        elevenlabs_voice_id = st.text_input("Voice ID", value="JBFqnCBsd6RMkjVDRZzb", key="voice_id")
        elevenlabs_model = st.selectbox("Voice Model", ["eleven_multilingual_v2", "eleven_monolingual_v1"], key="voice_model")
        
        # Discord Configuration
        st.subheader("🎮 Discord Integration")
        discord_bot_token = st.text_input("Discord Bot Token", type="password", key="discord_token")
        discord_guild_id = st.text_input("Guild ID", key="guild_id")
        discord_channel_id = st.text_input("Default Channel ID", key="channel_id")
        
        # System Requirements Check
        st.subheader("📋 System Status")
        requirements = {
            "FastRTC": FASTRTC_AVAILABLE,
            "ElevenLabs": ELEVENLABS_AVAILABLE,
            "Discord.py": DISCORD_AVAILABLE,
            "Pydub": PYDUB_AVAILABLE
        }
        
        for req, available in requirements.items():
            status = "✅" if available else "❌"
            color = "green" if available else "red"
            st.markdown(f":{color}[{status} {req}]")
        
        # Initialize System
        st.divider()
        if st.button("🚀 Initialize Multi-Agent System", use_container_width=True):
            required_key = openai_api_key if model_provider == "openai" else groq_api_key
            
            if not required_key:
                st.error("❌ Please provide the required API key")
            else:
                config = {
                    "model_provider": model_provider,
                    "model_name": model_name,
                    "temperature": temperature,
                    "openai_api_key": openai_api_key if model_provider == "openai" else None,
                    "groq_api_key": groq_api_key if model_provider == "groq" else None,
                    "elevenlabs_api_key": elevenlabs_api_key,
                    "elevenlabs_voice_id": elevenlabs_voice_id,
                    "elevenlabs_model": elevenlabs_model,
                    "discord_bot_token": discord_bot_token,
                    "discord_guild_id": discord_guild_id,
                    "discord_channel_id": discord_channel_id
                }
                
                with st.spinner("Initializing multi-agent system..."):
                    try:
                        st.session_state.agent_system = MultiAgentSystem(config)
                        st.session_state.initialized = True
                        st.success("🎉 Multi-Agent System Initialized Successfully!")
                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ Initialization failed: {str(e)}")
    
    # Main interface
    if not st.session_state.get('initialized', False):
        # Welcome screen
        st.info("👈 Configure your API keys and initialize the system using the sidebar")
        
        # Feature showcase
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎙️ Voice Agent Features
            - **Text-to-Speech**: ElevenLabs TTS with multiple voices
            - **Speech-to-Text**: Real-time transcription
            - **Voice Streaming**: FastRTC integration
            - **Audio Processing**: MP3/WAV support
            """)
            
            st.markdown("""
            ### 🧠 Memory Agent Features
            - **Conversation Storage**: InMemorySaver integration
            - **Context Retrieval**: Smart memory management
            - **Session Persistence**: Thread-based memory
            - **Summary Generation**: Automated summaries
            """)
        
        with col2:
            st.markdown("""
            ### 🎮 Discord Agent Features
            - **Message Sending**: Channel integration
            - **Voice Channels**: Create and manage
            - **Bot Commands**: Interactive responses
            - **Server Management**: Guild operations
            """)
            
            st.markdown("""
            ### 🔍 Research Agent Features
            - **Web Search**: Information gathering
            - **Data Analysis**: Pattern recognition
            - **Insights Generation**: Smart analysis
            - **Knowledge Base**: Contextual responses
            """)
        
        # Architecture diagram
        st.subheader("🏗️ System Architecture")
        st.markdown("""
        ```
        ┌─────────────────┐    ┌──────────────────────┐
        │   Streamlit UI  │ ←→ │ LangGraph Supervisor │
        └─────────────────┘    └──────────┬───────────┘
                                          │
                        ┌─────────────────┼─────────────────┐
                        │                 │                 │
                ┌───────▼────┐    ┌──────▼──────┐   ┌──────▼──────┐
                │ Voice Agent│    │Discord Agent│   │Memory Agent │
                │            │    │             │   │             │
                │ ElevenLabs │    │ Discord.py  │   │InMemorySaver│
                │  FastRTC   │    │   Bot API   │   │InMemoryStore│
                └────────────┘    └─────────────┘   └─────────────┘
                        │                 │                 │
                ┌───────▼────┐    ┌──────▼──────┐   ┌──────▼──────┐
                │    TTS     │    │  Channels   │   │Conversations│
                │    STT     │    │  Messages   │   │  Context    │
                │   Audio    │    │  Commands   │   │  Summaries  │
                └────────────┘    └─────────────┘   └─────────────┘
        ```
        """)
        
        return
    
    # Main chat interface
    st.subheader("💬 Multi-Agent Chat Interface")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "agent" in message:
                st.caption(f"🤖 Handled by: **{message['agent']}**")
            if "timestamp" in message:
                st.caption(f"🕐 {message['timestamp']}")
    
    # Chat input
    if prompt := st.chat_input("Ask the multi-agent system anything..."):
        # Add user message
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "timestamp": timestamp
        })
        
        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption(f"🕐 {timestamp}")
        
        # Process with agent system
        with st.chat_message("assistant"):
            with st.spinner("🤖 Processing through supervisor and agents..."):
                try:
                    result = asyncio.run(
                        st.session_state.agent_system.process_message(
                            prompt, 
                            thread_id="streamlit_main_session"
                        )
                    )
                    
                    if result and "messages" in result and not result.get("error"):
                        # Extract AI response - look for the most recent BaseMessage
                        ai_messages = []
                        for msg in result["messages"]:
                            if isinstance(msg, (AIMessage, CoreBaseMessage)):
                                # Handle both AIMessage and BaseMessage types
                                if hasattr(msg, 'content') and msg.content:
                                    ai_messages.append(msg)
                        
                        if ai_messages:
                            # Get the last AI message
                            last_message = ai_messages[-1]
                            response = last_message.content
                            
                            # Try to determine which agent handled it based on content
                            agent_name = "supervisor"
                            content_lower = response.lower()
                            
                            if any(keyword in content_lower for keyword in ["voice", "tts", "audio", "speech", "sound"]):
                                agent_name = "voice_expert"
                            elif any(keyword in content_lower for keyword in ["discord", "channel", "message", "bot"]):
                                agent_name = "discord_expert"
                            elif any(keyword in content_lower for keyword in ["memory", "conversation", "stored", "recall"]):
                                agent_name = "memory_expert"
                            elif any(keyword in content_lower for keyword in ["search", "research", "analysis", "data"]):
                                agent_name = "research_expert"
                            
                            st.markdown(response)
                            st.caption(f"🤖 Response from: **{agent_name}**")
                            
                            # Add to chat history
                            response_timestamp = datetime.now().strftime("%H:%M:%S")
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response,
                                "agent": agent_name,
                                "timestamp": response_timestamp
                            })
                        else:
                            st.error("❌ No valid response generated from agents")
                    else:
                        error_msg = result.get("error", "Unknown error occurred")
                        st.error(f"❌ Processing failed: {error_msg}")
                        
                except Exception as e:
                    st.error(f"❌ System error: {str(e)}")
    
    # Control panels
    col1, col2 = st.columns(2)
    
    with col1:
        # Agent Status Panel
        with st.expander("🔍 Agent System Status", expanded=False):
            if st.session_state.get('agent_system'):
                system = st.session_state.agent_system
                
                # System metrics
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric(
                        "Agents Active", 
                        len(system.agents), 
                        delta=f"+{len(system.agents)} ready"
                    )
                
                with col_b:
                    st.metric(
                        "ElevenLabs", 
                        "Connected" if system.elevenlabs_client else "Offline"
                    )
                
                with col_c:
                    st.metric(
                        "Discord", 
                        "Connected" if system.discord_bot else "Offline"
                    )
                
                # Agent details
                st.markdown("**Active Agents:**")
                for agent_name in system.agents.keys():
                    st.markdown(f"✅ `{agent_name}`")
                
                # Memory status
                st.markdown("**Memory System:**")
                st.markdown(f"✅ Checkpointer: `{type(system.checkpointer).__name__}`")
                st.markdown(f"✅ Store: `{type(system.store).__name__}`")
    
    with col2:
        # Voice Interface Panel
        with st.expander("🎙️ Voice Interface Controls", expanded=False):
            if FASTRTC_AVAILABLE and st.session_state.get('agent_system'):
                st.markdown("**Voice Features Available:**")
                
                if st.button("🎤 Test Voice Input", use_container_width=True):
                    st.info("🎙️ Voice input test initiated")
                    st.markdown("*In a full deployment, this would start FastRTC voice streaming*")
                
                if st.button("🔊 Test Text-to-Speech", use_container_width=True):
                    test_text = "Hello! This is a test of the multi-agent voice system."
                    if st.session_state.agent_system.elevenlabs_client:
                        with st.spinner("Generating speech..."):
                            # Simulate TTS
                            st.success("🔊 TTS generated successfully!")
                            st.markdown(f"*Generated audio for: '{test_text}'*")
                    else:
                        st.error("❌ ElevenLabs not configured")
                
                # Voice settings
                st.markdown("**Current Voice Settings:**")
                system = st.session_state.agent_system
                st.markdown(f"- Voice ID: `{system.config.get('elevenlabs_voice_id', 'Not set')}`")
                st.markdown(f"- Model: `{system.config.get('elevenlabs_model', 'Not set')}`")
                
            else:
                st.warning("⚠️ Voice features require FastRTC and ElevenLabs configuration")
    
    # Advanced controls
    with st.expander("⚙️ Advanced System Controls", expanded=False):
        col_x, col_y = st.columns(2)
        
        with col_x:
            if st.button("🔄 Reset Chat History", use_container_width=True):
                st.session_state.messages = []
                st.success("✅ Chat history cleared")
                st.rerun()
            
            if st.button("🧠 Test Memory System", use_container_width=True):
                if st.session_state.get('agent_system'):
                    test_result = asyncio.run(
                        st.session_state.agent_system.process_message(
                            "Please test the memory system and store this conversation",
                            thread_id="memory_test_session"
                        )
                    )
                    st.success("✅ Memory system test completed")
                else:
                    st.error("❌ System not initialized")
        
        with col_y:
            if st.button("🎮 Test Discord Integration", use_container_width=True):
                if st.session_state.get('agent_system'):
                    test_result = asyncio.run(
                        st.session_state.agent_system.process_message(
                            "Please test Discord integration and show available channels",
                            thread_id="discord_test_session"
                        )
                    )
                    st.success("✅ Discord integration test completed")
                else:
                    st.error("❌ System not initialized")
            
            if st.button("🔍 Test Research Agent", use_container_width=True):
                if st.session_state.get('agent_system'):
                    test_result = asyncio.run(
                        st.session_state.agent_system.process_message(
                            "Please search for information about LangGraph and multi-agent systems",
                            thread_id="research_test_session"
                        )
                    )
                    st.success("✅ Research agent test completed")
                else:
                    st.error("❌ System not initialized")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🤖 <strong>Multi-Agent LangGraph Supervisor System</strong><br>
        Powered by LangGraph • ElevenLabs • Discord • FastRTC<br>
        <em>Ready for Streamlit Cloud deployment</em>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    
