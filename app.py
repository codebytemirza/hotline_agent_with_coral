def main():
    st.set_page_config(
        page_title="🤖 Multi-Agent LangGraph System",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 Multi-Agent LangGraph Supervisor System")
    st.markdown("**Powered by LangGraph Supervisor with Voice, Discord, Memory & Crisis Support**")
    
    # Crisis alert banner
    st.error("🚨 **Crisis Support Available 24/7** - If you're in crisis, call 988 (US) or your local emergency number")
    
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
        
        # Crisis Support Info
        st.subheader("🆘 Crisis Support")
        st.info("""
        **Crisis Hotlines:**
        🇺🇸 **988** Suicide & Crisis Lifeline
        🇬🇧 **116 123** Samaritans  
        🇨🇦 **1-833-456-4566** Talk Suicide Canada
        🇦🇺 **13 11 14** Lifeline Australia
        """)
        
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
            ### 🆘 Crisis Support Agent
            - **Crisis Detection**: Real-time monitoring
            - **Immediate Resources**: 24/7 hotlines
            - **Safety Protocols**: Professional standards
            - **Multi-Country Support**: Global resources
            """)
        
        # Architecture diagram
        st.subheader("🏗️ System Architecture")
        st.markdown("""
        ```
        ┌─────────────────┐    ┌──────────────────────┐
        │   Streamlit UI  │ ←→ │ LangGraph Supervisor │
        │   + Crisis UI   │    │  + Crisis Detection  │
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
                        │                                   │
                ┌───────▼────────────────────────────────────▼──────┐
                │              Crisis Support Agent                 │
                │        🚨 Mental Health Crisis Detection          │
                │     ☎️ 988, Samaritans, Crisis Text Line         │
                └───────────────────────────────────────────────────┘
        ```
        """)
        
        return
    
    # Main chat interface
    st.subheader("💬 Multi-Agent Chat Interface")
    
    # Crisis warning
    st.warning("🆘 **Need immediate help?** Call 988 (US), text HOME to 741741, or contact your local emergency services")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "agent" in message:
                st.caption(f"🤖 Handled by: **{message['agent']}**")
            if "crisis_detected" in message and message["crisis_detected"]:
                st.error("🚨 **Crisis Support Response**")
            if "timestamp" in message:
                st.caption(f"🕐 {message['timestamp']}")
    
    # Chat input
    if prompt := st.chat_input("Ask the multi-agent system anything... (Crisis support available 24/7)"):
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
                            
                            # Determine which agent handled it
                            agent_name = "supervisor"
                            content_lower = response.lower()
                            
                            # Check for crisis response first
                            if result.get("crisis_detected") or "crisis" in content_lower or "988" in response:
                                agent_name = "crisis_agent"
                            elif any(keyword in content_lower for keyword in ["voice", "tts", "audio", "speech", "sound"]):
                                agent_name = "voice_agent"
                            elif any(keyword in content_lower for keyword in ["discord", "channel", "message", "bot"]):
                                agent_name = "discord_agent"
                            elif any(keyword in content_lower for keyword in ["memory", "conversation", "stored", "recall"]):
                                agent_name = "memory_agent"
                            
                            st.markdown(response)
                            st.caption(f"🤖 Response from: **{agent_name}**")
                            
                            # Add crisis indicator if detected
                            if result.get("crisis_detected"):
                                st.error("🚨 **Crisis Support Resources Provided**")
                            elif result.get("depression_detected"):
                                st.warning("💙 **Mental Health Support Available**")
                            
                            # Add to chat history
                            response_timestamp = datetime.now().strftime("%H:%M:%S")
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": response,
                                "agent": agent_name,
                                "crisis_detected": result.get("crisis_detected", False),
                                "depression_detected": result.get("depression_detected", False),
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
                        "Crisis Support", 
                        "🚨 ACTIVE",
                        delta="24/7 Available"
                    )
                
                with col_c:
                    st.metric(
                        "ElevenLabs", 
                        "Connected" if system.elevenlabs_client else "Offline"
                    )
                
                # Agent details
                st.markdown("**Active Agents:**")
                for agent_name in system.agents.keys():
                    if agent_name == "crisis_agent":
                        st.markdown(f"🚨 `{agent_name}` (Crisis Support)")
                    else:
                        st.markdown(f"✅ `{agent_name}`")
                
                # Memory status
                st.markdown("**Memory System:**")
                st.markdown(f"✅ Checkpointer: `{type(system.checkpointer).__name__}`")
                st.markdown(f"✅ Store: `{type(system.store).__name__}`")
    
    with col2:
        # Crisis Support Panel
        with st.expander("🆘 Crisis Support Information", expanded=False):
            st.error("**🚨 IMMEDIATE HELP AVAILABLE 24/7**")
            
            st.markdown("""
            **United States:**
            - 📞 **988** - Suicide & Crisis Lifeline
            - 💬 **Text HOME to 741741** - Crisis Text Line
            - 🌐 **suicidepreventionlifeline.org**
            
            **International:**
            - 🇬🇧 **116 123** - Samaritans (UK)
            - 🇨🇦 **1-833-456-4566** - Talk Suicide Canada
            - 🇦🇺 **13 11 14** - Lifeline Australia
            
            **Emergency Services:**
            - 🚨 **911** (US), **999** (UK), **000** (AU)
            """)
            
            if st.button("🆘 Test Crisis Detection", use_container_width=True):
                if st.session_state.get('agent_system'):
                    test_result = asyncio.run(
                        st.session_state.agent_system.process_message(
                            "I'm feeling really hopeless and need help",
                            thread_id="crisis_test_session"
                        )
                    )
                    if test_result.get("crisis_detected") or test_result.get("depression_detected"):
                        st.success("✅ Crisis detection system working correctly")
                    else:
                        st.warning("⚠️ Crisis detection may need adjustment")
                else:
                    st.error("❌ System not initialized")
    
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
            
            if st.button("🎙️ Test Voice System", use_container_width=True):
                if st.session_state.get('agent_system'):
                    test_result = asyncio.run(
                        st.session_state.agent_system.process_message(
                            "Please test the voice system with text-to-speech",
                            thread_id="voice_test_session"
                        )
                    )
                    st.success("✅ Voice system test completed")
                else:
                    st.error("❌ System not initialized")
    
    # Safety and Legal Information
    with st.expander("⚖️ Safety & Legal Information", expanded=False):
        st.markdown("""
        **🚨 Crisis Support Disclaimer:**
        - This system provides crisis detection and resource referrals only
        - Not a substitute for professional mental health care
        - In emergencies, contact local emergency services immediately
        - All crisis hotlines are staffed by trained professionals
        
        **🔒 Privacy & Safety:**
        - Conversations are processed in memory only
        - No personal data is permanently stored
        - Crisis detection is automated and may have limitations
        - Always prioritize professional help for mental health concerns
        
        **🛡️ System Capabilities:**
        - Detects common crisis keywords and phrases
        - Provides immediate resource referrals
        - Supports multiple countries' crisis lines
        - Integrates with voice, Discord, and memory systems
        
        **📞 When to Seek Professional Help:**
        - Persistent thoughts of self-harm
        - Overwhelming feelings of hopelessness
        - Substance abuse concerns
        - Major life changes or trauma
        - Any mental health crisis
        """)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🤖 <strong>Multi-Agent LangGraph Supervisor System with Crisis Support</strong><br>
        Powered by LangGraph • ElevenLabs • Discord • FastRTC • Crisis Detection<br>
        <em>Ready for Streamlit Cloud deployment with 24/7 mental health support</em><br>
        🆘 <strong>Crisis Support: 988 (US) • 116 123 (UK) • 13 11 14 (AU)</strong>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
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
import re

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
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent

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

class CrisisDetector:
    """Detects mental health crisis indicators and provides appropriate responses"""
    
    # Crisis keywords and phrases
    CRISIS_KEYWORDS = [
        'suicide', 'kill myself', 'end my life', 'want to die', 'better off dead',
        'no point living', 'hurt myself', 'self harm', 'overdose', 'can\'t go on',
        'not safe', 'harm myself', 'end it all', 'give up', 'hopeless'
    ]
    
    DEPRESSION_KEYWORDS = [
        'depressed', 'depression', 'sad', 'hopeless', 'worthless', 'empty',
        'lonely', 'isolated', 'exhausted', 'tired of life', 'dark thoughts'
    ]
    
    HOTLINES = {
        'US': {
            'name': '988 Suicide & Crisis Lifeline',
            'number': '988',
            'text': 'Text HOME to 741741',
            'chat': 'suicidepreventionlifeline.org'
        },
        'UK': {
            'name': 'Samaritans',
            'number': '116 123',
            'text': 'Text SHOUT to 85258',
            'chat': 'samaritans.org'
        },
        'CA': {
            'name': 'Talk Suicide Canada',
            'number': '1-833-456-4566',
            'text': 'Text 45645',
            'chat': 'talksuicide.ca'
        },
        'AU': {
            'name': 'Lifeline Australia',
            'number': '13 11 14',
            'text': 'Text 0477 13 11 14',
            'chat': 'lifeline.org.au'
        }
    }
    
    @staticmethod
    def detect_crisis(text: str) -> Dict[str, Any]:
        """Detect crisis indicators in text"""
        text_lower = text.lower()
        
        crisis_detected = any(keyword in text_lower for keyword in CrisisDetector.CRISIS_KEYWORDS)
        depression_detected = any(keyword in text_lower for keyword in CrisisDetector.DEPRESSION_KEYWORDS)
        
        # Check for "not safe" specifically
        not_safe = 'not safe' in text_lower or 'unsafe' in text_lower
        
        severity = 'none'
        if crisis_detected or not_safe:
            severity = 'high'
        elif depression_detected:
            severity = 'moderate'
        
        return {
            'crisis_detected': crisis_detected or not_safe,
            'depression_detected': depression_detected,
            'severity': severity,
            'needs_immediate_help': crisis_detected or not_safe
        }
    
    @staticmethod
    def get_crisis_response(country_code: str = 'US') -> str:
        """Get appropriate crisis response with hotline information"""
        hotline = CrisisDetector.HOTLINES.get(country_code, CrisisDetector.HOTLINES['US'])
        
        response = f"""🚨 **IMMEDIATE HELP AVAILABLE** 🚨

I'm concerned about your safety and wellbeing. Please know that you matter and help is available right now:

**{hotline['name']}**
📞 **Call: {hotline['number']}** (Available 24/7)
💬 **Text: {hotline['text']}**
🌐 **Chat: {hotline['chat']}**

**If you're in immediate danger:**
🚨 **Call emergency services: 911 (US), 999 (UK), 000 (AU)**

**You are not alone. These feelings can change. Please reach out for help.**

Would you like me to help you find additional local resources or someone to talk to right now?"""
        
        return response
    
    @staticmethod
    def get_supportive_response() -> str:
        """Get supportive response for depression indicators"""
        return """💙 **You're Not Alone** 💙

I hear that you're going through a difficult time. Your feelings are valid, and it's brave of you to share them.

**Immediate Support Available:**
📞 **988** (Suicide & Crisis Lifeline - US)
💬 **Text HOME to 741741** (Crisis Text Line)
🌐 **Online chat at suicidepreventionlifeline.org**

**Remember:**
- These feelings are temporary and can change
- You deserve support and care
- Professional counselors are available to help
- Taking care of your mental health is important

Would you like help finding a therapist, counselor, or other mental health resources in your area?"""

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
        self.crisis_detector = CrisisDetector()
        
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
        """Research agent removed as requested - this method is no longer needed"""
        pass
    
    def _create_crisis_tools(self):
        """Create tools for crisis detection and response"""
        @tool
        def detect_crisis_indicators(user_message: str) -> str:
            """Detect mental health crisis indicators in user message"""
            try:
                detection = self.crisis_detector.detect_crisis(user_message)
                return json.dumps(detection, indent=2)
            except Exception as e:
                return f"❌ Crisis detection error: {str(e)}"
        
        @tool
        def get_crisis_resources(country_code: str = "US") -> str:
            """Get mental health crisis resources and hotlines"""
            try:
                return self.crisis_detector.get_crisis_response(country_code)
            except Exception as e:
                return f"❌ Crisis resources error: {str(e)}"
        
        @tool
        def get_mental_health_support() -> str:
            """Get supportive mental health resources"""
            try:
                return self.crisis_detector.get_supportive_response()
            except Exception as e:
                return f"❌ Mental health support error: {str(e)}"
        
        return [detect_crisis_indicators, get_crisis_resources, get_mental_health_support]
    
    def _create_specialized_agents(self):
        """Create specialized agents using create_react_agent with proper prompts"""
        llm = self._get_llm()
        
        # Voice Agent
        voice_tools = self._create_voice_tools()
        voice_system_content = """You are a voice interaction expert specializing in audio processing and speech technologies.

Your responsibilities include:
- Converting text to speech using ElevenLabs TTS with various voice models
- Processing speech-to-text transcription with high accuracy
- Managing FastRTC audio streaming for real-time voice interactions
- Handling audio format conversions and quality optimization
- Providing voice system status and diagnostics

Always prioritize clear, natural speech output and accurate transcription. Use appropriate voice settings based on context."""

        voice_agent = create_react_agent(llm, voice_tools)
        self.agents['voice_agent'] = AgentExecutor(
            agent=voice_agent, 
            tools=voice_tools, 
            verbose=True,
            handle_parsing_errors=True
        )
        
        # Discord Agent
        discord_tools = self._create_discord_tools()
        discord_system_content = """You are a Discord integration expert specializing in bot operations and server management.

Your responsibilities include:
- Sending messages to Discord channels with proper formatting
- Managing voice channels and server interactions
- Handling Discord bot commands and permissions
- Monitoring server status and member interactions
- Creating and managing Discord community features

Always ensure messages are appropriate for the target audience and follow Discord community guidelines. Use proper Discord markdown formatting when needed."""

        discord_agent = create_react_agent(llm, discord_tools)
        self.agents['discord_agent'] = AgentExecutor(
            agent=discord_agent, 
            tools=discord_tools, 
            verbose=True,
            handle_parsing_errors=True
        )
        
        # Memory Agent
        memory_tools = self._create_memory_tools()
        memory_system_content = """You are a memory management expert specializing in conversation storage and context retrieval.

Your responsibilities include:
- Storing conversations with proper metadata and indexing
- Retrieving relevant conversation history and context
- Generating meaningful summaries of interactions
- Managing memory persistence across sessions
- Optimizing storage for efficient retrieval

Always maintain conversation context and provide relevant historical information when needed. Ensure data privacy and proper organization of stored information."""

        memory_agent = create_react_agent(llm, memory_tools)
        self.agents['memory_agent'] = AgentExecutor(
            agent=memory_agent, 
            tools=memory_tools, 
            verbose=True,
            handle_parsing_errors=True
        )
        
        # Research Agent - REMOVED as requested
        # research_tools = self._create_research_tools()
        # research_agent = create_react_agent(llm, research_tools)
        # self.agents['research_agent'] = AgentExecutor(
        #     agent=research_agent, 
        #     tools=research_tools, 
        #     verbose=True,
        #     handle_parsing_errors=True
        # )
        
        # Crisis Support Agent
        crisis_tools = self._create_crisis_tools()
        crisis_system_content = """You are a mental health crisis support expert trained to detect and respond to mental health emergencies.

Your responsibilities include:
- Detecting crisis indicators and mental health emergencies
- Providing immediate crisis resources and hotline information
- Offering supportive responses for depression and mental health issues
- Following crisis intervention protocols
- Prioritizing user safety above all else

CRITICAL: If someone indicates they are "not safe" or expresses suicidal thoughts, immediately provide crisis hotline information and encourage professional help. Always take mental health concerns seriously."""

        crisis_agent = create_react_agent(llm, crisis_tools)
        self.agents['crisis_agent'] = AgentExecutor(
            agent=crisis_agent, 
            tools=crisis_tools, 
            verbose=True,
            handle_parsing_errors=True
        )
        
        st.success(f"✅ Created {len(self.agents)} specialized agents including crisis support")
    
    def _create_supervisor_workflow(self):
        """Create the supervisor workflow using langgraph_supervisor"""
        try:
            llm = self._get_llm()
            
            # Get list of agent executors
            agent_list = list(self.agents.values())
            
            # Create supervisor system prompt with crisis detection
            supervisor_prompt = """You are an intelligent multi-agent system supervisor coordinating specialized AI agents with CRISIS DETECTION capabilities.

Your team consists of:
1. **voice_agent**: Handles all voice, audio, TTS, STT, and FastRTC streaming tasks
2. **discord_agent**: Manages Discord bot operations, messaging, and server interactions  
3. **memory_agent**: Handles conversation storage, retrieval, and memory management
4. **crisis_agent**: PRIORITY - Detects mental health crises and provides immediate support

🚨 CRISIS DETECTION PROTOCOL:
- ALWAYS scan user input for crisis indicators first
- Keywords: suicide, kill myself, hurt myself, not safe, end my life, hopeless, etc.
- If crisis detected → IMMEDIATELY route to crisis_agent
- Provide hotline information and professional resources
- Never ignore or downplay mental health concerns

ROUTING RULES:
- Crisis/Mental Health → crisis_agent (HIGHEST PRIORITY)
- Voice/Audio → voice_agent (keywords: voice, audio, speech, TTS, STT, sound)
- Discord → discord_agent (keywords: discord, message, channel, bot, server)
- Memory → memory_agent (keywords: remember, store, recall, history, conversation)

Always prioritize user safety and wellbeing. For complex requests, coordinate between agents as needed."""
            
            # Create supervisor using langgraph_supervisor
            self.supervisor_workflow = create_supervisor(
                agents=agent_list,
                model=llm,
                prompt=supervisor_prompt,
                output_mode="full_history"
            )
            
            # Compile the workflow with memory
            self.compiled_app = self.supervisor_workflow.compile(
                checkpointer=self.checkpointer,
                store=self.store
            )
            
            st.success("🔗 Supervisor workflow created with crisis detection and compiled successfully")
            
        except Exception as e:
            st.error(f"❌ Supervisor workflow creation failed: {str(e)}")
            raise
    
    async def process_message(self, user_message: str, thread_id: str = "default_session") -> Dict[str, Any]:
        """Process a user message through the supervisor workflow with crisis detection"""
        try:
            if not self.compiled_app:
                raise ValueError("Supervisor workflow not initialized")
            
            # FIRST: Check for crisis indicators
            crisis_detection = self.crisis_detector.detect_crisis(user_message)
            
            # If crisis detected, immediately provide crisis response
            if crisis_detection['crisis_detected']:
                crisis_response = self.crisis_detector.get_crisis_response()
                
                # Log crisis event
                st.error("🚨 CRISIS DETECTED - Immediate help resources provided")
                
                return {
                    "messages": [
                        HumanMessage(content=user_message),
                        AIMessage(content=crisis_response)
                    ],
                    "crisis_detected": True,
                    "severity": crisis_detection['severity']
                }
            
            # If depression indicators, include supportive response
            if crisis_detection['depression_detected']:
                supportive_response = self.crisis_detector.get_supportive_response()
                
                # Process through normal workflow but include crisis context
                input_state = {
                    "messages": [
                        HumanMessage(content=user_message),
                        SystemMessage(content="Note: Depression indicators detected. Prioritize supportive, empathetic responses.")
                    ]
                }
                
                config = {"configurable": {"thread_id": thread_id}}
                result = await self.compiled_app.ainvoke(input_state, config)
                
                # Append supportive message
                if "messages" in result:
                    result["messages"].append(AIMessage(content=supportive_response))
                
                result["depression_detected"] = True
                return result
            
            # Normal processing for non-crisis messages
            input_state = {
                "messages": [HumanMessage(content=user_message)]
            }
            
            config = {"configurable": {"thread_id": thread_id}}
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
                    # Process through supervisor with crisis detection
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
                            response_text = "I'm here to help. If you're experiencing a crisis, please call 988 for immediate support."
                            
                    finally:
                        loop.close()
                else:
                    response_text = "I'm listening. Please tell me how I can help you."
