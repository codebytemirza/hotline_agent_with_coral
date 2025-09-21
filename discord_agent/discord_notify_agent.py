import streamlit as st
import discord
import asyncio
import os
import json
import traceback
import sqlite3
import pandas as pd
from datetime import datetime
from typing import Optional, Type, Any, List, Dict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import BaseTool
from langchain.schema import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="🆘 Discord Safety Hotline Agent",
    page_icon="🆘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #ff4444;
    font-size: 2.5rem;
    margin-bottom: 1rem;
}
.safety-alert {
    background-color: #ffebee;
    border: 2px solid #ff4444;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
}
.chat-container {
    max-height: 500px;
    overflow-y: auto;
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 1rem;
    margin: 1rem 0;
}
.user-message {
    background-color: #e3f2fd;
    padding: 0.5rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}
.agent-message {
    background-color: #f1f8e9;
    padding: 0.5rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Database Manager Class
class DatabaseManager:
    def __init__(self, db_path: str = "hotline_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_message TEXT NOT NULL,
                agent_response TEXT NOT NULL,
                safety_level TEXT DEFAULT 'low',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_ip TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # User details table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_details (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                name TEXT,
                phone TEXT,
                location TEXT,
                email TEXT,
                age INTEGER,
                emergency_contact TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_name TEXT,
                phone TEXT,
                location TEXT,
                concern_type TEXT,
                urgency_level TEXT,
                details TEXT,
                discord_sent BOOLEAN DEFAULT FALSE,
                discord_message_id TEXT,
                status TEXT DEFAULT 'pending',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                resolved_at DATETIME,
                notes TEXT
            )
        """)
        
        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_messages INTEGER DEFAULT 0,
                safety_alerts INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active'
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_conversation(self, session_id: str, user_message: str, 
                         agent_response: str, safety_level: str = "low", 
                         user_ip: str = None):
        """Save conversation to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO conversations 
            (session_id, user_message, agent_response, safety_level, user_ip)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, user_message, agent_response, safety_level, user_ip))
        
        # Update session activity
        cursor.execute("""
            INSERT OR REPLACE INTO sessions 
            (session_id, last_activity, total_messages)
            VALUES (?, CURRENT_TIMESTAMP, 
                    COALESCE((SELECT total_messages FROM sessions WHERE session_id = ?), 0) + 1)
        """, (session_id, session_id))
        
        conn.commit()
        conn.close()
    
    def save_user_details(self, session_id: str, details: Dict):
        """Save user details to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO user_details 
            (session_id, name, phone, location, email, age, emergency_contact)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id, 
            details.get('name'),
            details.get('phone'),
            details.get('location'),
            details.get('email'),
            details.get('age'),
            details.get('emergency_contact')
        ))
        
        conn.commit()
        conn.close()
    
    def save_alert(self, session_id: str, alert_data: Dict) -> int:
        """Save alert to database and return alert ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO alerts 
            (session_id, user_name, phone, location, concern_type, 
             urgency_level, details, discord_sent, discord_message_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            alert_data.get('user_name'),
            alert_data.get('phone'),
            alert_data.get('location'),
            alert_data.get('concern_type'),
            alert_data.get('urgency'),
            alert_data.get('details'),
            alert_data.get('discord_sent', False),
            alert_data.get('discord_message_id')
        ))
        
        alert_id = cursor.lastrowid
        
        # Update session safety alerts count
        cursor.execute("""
            UPDATE sessions 
            SET safety_alerts = COALESCE(safety_alerts, 0) + 1
            WHERE session_id = ?
        """, (session_id,))
        
        conn.commit()
        conn.close()
        
        return alert_id
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Retrieve conversation history for a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT user_message, agent_response, safety_level, timestamp
            FROM conversations 
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """, (session_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                "user_message": row[0],
                "agent_response": row[1],
                "safety_level": row[2],
                "timestamp": row[3]
            }
            for row in results
        ]
    
    def get_user_details(self, session_id: str) -> Dict:
        """Get user details for a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name, phone, location, email, age, emergency_contact
            FROM user_details 
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
        """, (session_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                "name": result[0],
                "phone": result[1],
                "location": result[2],
                "email": result[3],
                "age": result[4],
                "emergency_contact": result[5]
            }
        return {}
    
    def get_all_alerts(self, limit: int = 50) -> pd.DataFrame:
        """Get all alerts as DataFrame"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT id, session_id, user_name, phone, location, concern_type,
                   urgency_level, details, discord_sent, status, created_at
            FROM alerts 
            ORDER BY created_at DESC
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        
        return df
    
    def update_alert_status(self, alert_id: int, status: str, notes: str = None):
        """Update alert status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if status == 'resolved':
            cursor.execute("""
                UPDATE alerts 
                SET status = ?, notes = ?, resolved_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (status, notes, alert_id))
        else:
            cursor.execute("""
                UPDATE alerts 
                SET status = ?, notes = ?
                WHERE id = ?
            """, (status, notes, alert_id))
        
        conn.commit()
        conn.close()
    
    def get_session_stats(self) -> Dict:
        """Get overall statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total sessions
        cursor.execute("SELECT COUNT(*) FROM sessions")
        total_sessions = cursor.fetchone()[0]
        
        # Total conversations
        cursor.execute("SELECT COUNT(*) FROM conversations")
        total_messages = cursor.fetchone()[0]
        
        # Total alerts
        cursor.execute("SELECT COUNT(*) FROM alerts")
        total_alerts = cursor.fetchone()[0]
        
        # Critical alerts
        cursor.execute("SELECT COUNT(*) FROM alerts WHERE urgency_level = 'critical'")
        critical_alerts = cursor.fetchone()[0]
        
        # Pending alerts
        cursor.execute("SELECT COUNT(*) FROM alerts WHERE status = 'pending'")
        pending_alerts = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "total_alerts": total_alerts,
            "critical_alerts": critical_alerts,
            "pending_alerts": pending_alerts
        }

# Initialize session state
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_details" not in st.session_state:
        st.session_state.user_details = {}
    if "safety_concern_detected" not in st.session_state:
        st.session_state.safety_concern_detected = False
    if "agent_executor" not in st.session_state:
        st.session_state.agent_executor = None
    if "collecting_details" not in st.session_state:
        st.session_state.collecting_details = False
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(datetime.now()))}"
    if "db_manager" not in st.session_state:
        st.session_state.db_manager = DatabaseManager()

init_session_state()

# Discord Tool Classes
class SendDiscordAlertInput(BaseModel):
    user_name: str = Field(description="Name of the user in crisis")
    location: str = Field(description="Location/address of the user")
    phone: str = Field(description="Phone number of the user")
    concern_type: str = Field(description="Type of safety concern")
    details: str = Field(description="Additional details about the situation")
    urgency: str = Field(description="Urgency level: low, medium, high, critical")

class SendDiscordAlertTool(BaseTool):
    name: str = "send_discord_alert"
    description: str = "Send a safety alert to Discord channel when user is in danger"
    args_schema: Type[BaseModel] = SendDiscordAlertInput
    
    def __init__(self, bot_token: str, channel_id: str, db_manager: DatabaseManager):
        super().__init__()
        self.bot_token = bot_token
        self.channel_id = int(channel_id)
        self.db_manager = db_manager
        
    async def _arun(self, user_name: str, location: str, phone: str, 
                   concern_type: str, details: str, urgency: str) -> str:
        try:
            # Create Discord client
            intents = discord.Intents.default()
            intents.message_content = True
            client = discord.Client(intents=intents)
            
            alert_data = {
                'user_name': user_name,
                'phone': phone,
                'location': location,
                'concern_type': concern_type,
                'urgency': urgency,
                'details': details,
                'discord_sent': False,
                'discord_message_id': None
            }
            
            @client.event
            async def on_ready():
                try:
                    channel = client.get_channel(self.channel_id)
                    if not channel:
                        st.error(f"Channel {self.channel_id} not found!")
                        return
                    
                    # Create emergency alert embed
                    embed = discord.Embed(
                        title="🚨 SAFETY ALERT - IMMEDIATE ATTENTION REQUIRED",
                        color=0xff0000 if urgency == "critical" else 0xff8800,
                        timestamp=datetime.utcnow()
                    )
                    
                    embed.add_field(name="👤 Name", value=user_name, inline=True)
                    embed.add_field(name="📍 Location", value=location, inline=True)
                    embed.add_field(name="📞 Phone", value=phone, inline=True)
                    embed.add_field(name="⚠️ Concern Type", value=concern_type, inline=True)
                    embed.add_field(name="🔥 Urgency", value=urgency.upper(), inline=True)
                    embed.add_field(name="📝 Details", value=details, inline=False)
                    embed.add_field(name="🆔 Session", value=st.session_state.session_id, inline=True)
                    
                    embed.set_footer(text="Discord Safety Hotline Agent")
                    
                    # Send alert message
                    alert_message = f"@everyone **SAFETY ALERT** - Immediate response needed!"
                    message = await channel.send(content=alert_message, embed=embed)
                    
                    # Update alert data with Discord info
                    alert_data['discord_sent'] = True
                    alert_data['discord_message_id'] = str(message.id)
                    
                    # Save alert to database
                    alert_id = self.db_manager.save_alert(st.session_state.session_id, alert_data)
                    
                    st.success(f"✅ Emergency alert sent to Discord successfully! Alert ID: {alert_id}")
                    
                    # Save user details to database
                    user_details = {
                        'name': user_name,
                        'phone': phone,
                        'location': location
                    }
                    self.db_manager.save_user_details(st.session_state.session_id, user_details)
                    
                except Exception as e:
                    st.error(f"Error sending alert: {str(e)}")
                    # Still save to database even if Discord fails
                    alert_data['discord_sent'] = False
                    self.db_manager.save_alert(st.session_state.session_id, alert_data)
                finally:
                    await client.close()
            
            # Start client
            await client.start(self.bot_token)
            return f"Alert sent successfully for session {st.session_state.session_id}"
            
        except Exception as e:
            error_msg = f"Failed to send Discord alert: {str(e)}"
            st.error(error_msg)
            # Save failed alert to database
            alert_data['discord_sent'] = False
            self.db_manager.save_alert(st.session_state.session_id, alert_data)
            return error_msg
    
    def _run(self, *args, **kwargs) -> str:
        return "Sync version not implemented - use async version"

class SafetyAssessmentTool(BaseTool):
    name: str = "assess_safety_concern"
    description: str = "Assess if user message contains safety concerns and determine urgency level"
    args_schema: Type[BaseModel] = BaseModel
    
    def _run(self, message: str) -> str:
        # Safety keywords and phrases
        critical_keywords = [
            "suicide", "kill myself", "end my life", "want to die", 
            "hurt myself", "self harm", "overdose", "gun", "knife"
        ]
        
        high_keywords = [
            "abuse", "violence", "threat", "danger", "scared", 
            "help me", "emergency", "crisis", "unsafe"
        ]
        
        medium_keywords = [
            "depressed", "anxious", "panic", "stressed", "overwhelmed",
            "hopeless", "alone", "struggling"
        ]
        
        message_lower = message.lower()
        
        if any(keyword in message_lower for keyword in critical_keywords):
            return "critical"
        elif any(keyword in message_lower for keyword in high_keywords):
            return "high"
        elif any(keyword in message_lower for keyword in medium_keywords):
            return "medium"
        else:
            return "low"

# Safety Agent Class
class SafetyHotlineAgent:
    def __init__(self, model_provider: str, model_name: str, api_key: str, 
                 discord_token: str, channel_id: str, db_manager: DatabaseManager, temperature: float = 0.7):
        
        # Initialize LLM
        self.model = init_chat_model(
            model=model_name,
            model_provider=model_provider,
            api_key=api_key,
            temperature=temperature,
            max_tokens=1000
        )
        
        self.db_manager = db_manager
        
        # Initialize tools
        self.discord_tool = SendDiscordAlertTool(discord_token, channel_id, db_manager)
        self.safety_tool = SafetyAssessmentTool()
        self.tools = [self.discord_tool, self.safety_tool]
        
        # Create agent prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a compassionate AI safety hotline agent. Your mission is to help people in crisis.

CRITICAL INSTRUCTIONS:
1. ALWAYS prioritize user safety above all else
2. Listen empathetically and respond with care and understanding
3. If you detect ANY safety concern, ask for user details:
   - Full name
   - Current location/address
   - Phone number
   - Nature of the emergency/concern
4. When you have collected user details AND detected a safety concern, use the send_discord_alert tool
5. Provide crisis resources and encourage professional help
6. Never dismiss or minimize someone's concerns
7. If someone mentions self-harm, violence, or being in danger, treat it seriously

ASSESSMENT LEVELS:
- CRITICAL: Immediate danger, self-harm, suicide ideation
- HIGH: Abuse, violence, threats, immediate safety concerns  
- MEDIUM: Mental health crisis, severe distress
- LOW: General support, non-urgent concerns

Always respond with empathy and provide appropriate resources."""),
            ("placeholder", "{agent_scratchpad}"),
            ("human", "{input}")
        ])
        
        # Create agent
        self.agent = create_tool_calling_agent(self.model, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent, 
            tools=self.tools, 
            verbose=True,
            handle_parsing_errors=True
        )
    
    async def process_message(self, user_message: str) -> str:
        try:
            # Assess safety level
            safety_level = self.safety_tool._run(user_message)
            
            response = await self.agent_executor.ainvoke({
                "input": user_message,
                "agent_scratchpad": []
            })
            
            agent_response = response.get("output", "I'm here to help. Please tell me more.")
            
            # Save conversation to database
            self.db_manager.save_conversation(
                st.session_state.session_id, 
                user_message, 
                agent_response, 
                safety_level
            )
            
            return agent_response
            
        except Exception as e:
            st.error(f"Agent error: {str(e)}")
            error_response = "I'm sorry, I'm having technical difficulties. Please try again or contact emergency services if this is urgent."
            
            # Still save the conversation even if there's an error
            self.db_manager.save_conversation(
                st.session_state.session_id, 
                user_message, 
                error_response, 
                "error"
            )
            
            return error_response

# Sidebar Configuration
def render_sidebar():
    st.sidebar.title("🔧 Configuration")
    
    # Add navigation tabs in sidebar
    tab = st.sidebar.radio(
        "Navigation", 
        ["🏠 Chat", "📊 Dashboard", "📋 Alerts", "💾 Data Export"], 
        key="nav_tab"
    )
    
    with st.sidebar.expander("🤖 AI Model Settings", expanded=True):
        model_provider = st.selectbox(
            "Model Provider",
            options=["openai", "groq", "anthropic", "ollama"],
            index=0
        )
        
        model_name = st.text_input(
            "Model Name", 
            value="gpt-3.5-turbo" if model_provider == "openai" else "llama3-8b-8192",
            help="e.g., gpt-3.5-turbo, llama3-8b-8192, claude-3-sonnet"
        )
        
        api_key = st.text_input(
            "API Key", 
            type="password",
            help="Your AI model API key"
        )
        
        temperature = st.slider(
            "Temperature", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.7,
            help="Controls response creativity"
        )
    
    with st.sidebar.expander("📱 Discord Settings", expanded=True):
        discord_token = st.text_input(
            "Discord Bot Token", 
            type="password",
            help="Your Discord bot token"
        )
        
        channel_id = st.text_input(
            "Discord Channel ID",
            help="Channel ID where alerts will be sent"
        )
    
    # Show current session info
    with st.sidebar.expander("📈 Session Info", expanded=False):
        st.write(f"**Session ID:** `{st.session_state.session_id}`")
        st.write(f"**Messages:** {len(st.session_state.messages)}")
        if st.session_state.safety_concern_detected:
            st.warning("⚠️ Safety concern detected")
    
    # Database controls
    with st.sidebar.expander("💾 Data Management", expanded=False):
        if st.button("🔄 Load Session History"):
            load_session_history()
        
        if st.button("🗑️ Clear Current Session"):
            clear_current_session()
        
        if st.button("📤 Export Session Data"):
            export_session_data()
    
    with st.sidebar.expander("📋 Emergency Resources", expanded=False):
        st.markdown("""
        **🚨 Emergency Numbers:**
        - 911 (US Emergency)
        - 988 (US Suicide Hotline)
        - 1-800-273-8255 (Crisis Line)
        
        **🌐 Online Resources:**
        - Crisis Text Line: Text HOME to 741741
        - National Suicide Prevention Lifeline
        - SAMHSA Helpline: 1-800-662-4357
        """)
    
    return model_provider, model_name, api_key, temperature, discord_token, channel_id, tab

# Helper functions for data management
def load_session_history():
    """Load conversation history from database"""
    try:
        history = st.session_state.db_manager.get_conversation_history(st.session_state.session_id)
        st.session_state.messages = []
        
        for conv in history:
            st.session_state.messages.append({
                "role": "user",
                "content": conv["user_message"],
                "timestamp": conv["timestamp"]
            })
            st.session_state.messages.append({
                "role": "agent", 
                "content": conv["agent_response"],
                "timestamp": conv["timestamp"]
            })
        
        # Load user details
        user_details = st.session_state.db_manager.get_user_details(st.session_state.session_id)
        st.session_state.user_details.update(user_details)
        
        st.success(f"Loaded {len(history)} conversations from database")
    except Exception as e:
        st.error(f"Error loading session history: {str(e)}")

def clear_current_session():
    """Clear current session data"""
    st.session_state.messages = []
    st.session_state.user_details = {}
    st.session_state.safety_concern_detected = False
    st.session_state.collecting_details = False
    st.success("Current session cleared")

def export_session_data():
    """Export current session data"""
    try:
        session_data = {
            "session_id": st.session_state.session_id,
            "messages": st.session_state.messages,
            "user_details": st.session_state.user_details,
            "conversation_history": st.session_state.db_manager.get_conversation_history(st.session_state.session_id)
        }
        
        json_data = json.dumps(session_data, indent=2, default=str)
        
        st.download_button(
            label="📥 Download Session Data",
            data=json_data,
            file_name=f"session_{st.session_state.session_id}.json",
            mime="application/json"
        )
    except Exception as e:
        st.error(f"Error exporting session data: {str(e)}")

# Dashboard functions
def render_dashboard():
    """Render analytics dashboard"""
    st.header("📊 Safety Hotline Dashboard")
    
    try:
        # Get statistics
        stats = st.session_state.db_manager.get_session_stats()
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sessions", stats["total_sessions"])
        
        with col2:
            st.metric("Total Messages", stats["total_messages"])
        
        with col3:
            st.metric("Total Alerts", stats["total_alerts"])
        
        with col4:
            st.metric("Critical Alerts", stats["critical_alerts"], 
                     delta=stats["pending_alerts"], delta_color="inverse")
        
        # Recent alerts chart
        st.subheader("📈 Recent Activity")
        
        alerts_df = st.session_state.db_manager.get_all_alerts(limit=20)
        
        if not alerts_df.empty:
            # Convert timestamp
            alerts_df['created_at'] = pd.to_datetime(alerts_df['created_at'])
            
            # Group by urgency level
            urgency_counts = alerts_df['urgency_level'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔥 Alerts by Urgency")
                st.bar_chart(urgency_counts)
            
            with col2:
                st.subheader("📊 Alert Status")
                status_counts = alerts_df['status'].value_counts()
                st.bar_chart(status_counts)
            
            # Recent alerts table
            st.subheader("🚨 Recent Alerts")
            display_df = alerts_df[['created_at', 'user_name', 'urgency_level', 'concern_type', 'status']].copy()
            display_df['created_at'] = display_df['created_at'].dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(display_df, use_container_width=True)
        
        else:
            st.info("No alerts recorded yet.")
    
    except Exception as e:
        st.error(f"Error loading dashboard: {str(e)}")

def render_alerts_management():
    """Render alerts management interface"""
    st.header("📋 Alert Management")
    
    try:
        alerts_df = st.session_state.db_manager.get_all_alerts(limit=100)
        
        if not alerts_df.empty:
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                status_filter = st.selectbox(
                    "Filter by Status",
                    options=['All'] + alerts_df['status'].unique().tolist()
                )
            
            with col2:
                urgency_filter = st.selectbox(
                    "Filter by Urgency", 
                    options=['All'] + alerts_df['urgency_level'].unique().tolist()
                )
            
            with col3:
                discord_filter = st.selectbox(
                    "Discord Status",
                    options=['All', 'Sent', 'Failed']
                )
            
            # Apply filters
            filtered_df = alerts_df.copy()
            
            if status_filter != 'All':
                filtered_df = filtered_df[filtered_df['status'] == status_filter]
            
            if urgency_filter != 'All':
                filtered_df = filtered_df[filtered_df['urgency_level'] == urgency_filter]
            
            if discord_filter == 'Sent':
                filtered_df = filtered_df[filtered_df['discord_sent'] == True]
            elif discord_filter == 'Failed':
                filtered_df = filtered_df[filtered_df['discord_sent'] == False]
            
            # Display alerts
            st.subheader(f"📊 Showing {len(filtered_df)} alerts")
            
            for idx, alert in filtered_df.iterrows():
                with st.expander(f"🚨 Alert #{alert['id']} - {alert['urgency_level'].upper()} - {alert['created_at']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Name:** {alert['user_name']}")
                        st.write(f"**Phone:** {alert['phone']}")
                        st.write(f"**Location:** {alert['location']}")
                        st.write(f"**Concern:** {alert['concern_type']}")
                        st.write(f"**Session:** {alert['session_id']}")
                    
                    with col2:
                        st.write(f"**Status:** {alert['status']}")
                        st.write(f"**Urgency:** {alert['urgency_level']}")
                        st.write(f"**Discord:** {'✅ Sent' if alert['discord_sent'] else '❌ Failed'}")
                        
                        # Status update
                        new_status = st.selectbox(
                            "Update Status",
                            options=['pending', 'in_progress', 'resolved', 'false_alarm'],
                            index=['pending', 'in_progress', 'resolved', 'false_alarm'].index(alert['status']),
                            key=f"status_{alert['id']}"
                        )
                        
                        notes = st.text_area(
                            "Notes", 
                            value="",
                            key=f"notes_{alert['id']}"
                        )
                        
                        if st.button(f"Update Alert #{alert['id']}", key=f"update_{alert['id']}"):
                            st.session_state.db_manager.update_alert_status(
                                alert['id'], new_status, notes
                            )
                            st.success("Alert updated!")
                            st.rerun()
                    
                    st.write(f"**Details:** {alert['details']}")
        
        else:
            st.info("No alerts found.")
    
    except Exception as e:
        st.error(f"Error loading alerts: {str(e)}")

def render_data_export():
    """Render data export interface"""
    st.header("💾 Data Export")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Export Options")
        
        export_type = st.radio(
            "Select data to export:",
            options=[
                "All Conversations",
                "All Alerts", 
                "Session Statistics",
                "User Details",
                "Complete Database"
            ]
        )
        
        date_range = st.date_input(
            "Date Range",
            value=[datetime.now().date(), datetime.now().date()],
            help="Select date range for export"
        )
        
        if st.button("🚀 Generate Export", type="primary"):
            try:
                if export_type == "All Conversations":
                    # Export conversations
                    conn = sqlite3.connect(st.session_state.db_manager.db_path)
                    df = pd.read_sql_query(
                        "SELECT * FROM conversations WHERE DATE(timestamp) BETWEEN ? AND ?",
                        conn, params=(date_range[0], date_range[1])
                    )
                    conn.close()
                    
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Conversations CSV",
                        data=csv_data,
                        file_name=f"conversations_{date_range[0]}_{date_range[1]}.csv",
                        mime="text/csv"
                    )
                
                elif export_type == "All Alerts":
                    # Export alerts
                    conn = sqlite3.connect(st.session_state.db_manager.db_path)
                    df = pd.read_sql_query(
                        "SELECT * FROM alerts WHERE DATE(created_at) BETWEEN ? AND ?",
                        conn, params=(date_range[0], date_range[1])
                    )
                    conn.close()
                    
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Alerts CSV",
                        data=csv_data,
                        file_name=f"alerts_{date_range[0]}_{date_range[1]}.csv",
                        mime="text/csv"
                    )
                
                elif export_type == "Complete Database":
                    # Export entire database as JSON
                    conn = sqlite3.connect(st.session_state.db_manager.db_path)
                    
                    data = {
                        "conversations": pd.read_sql_query("SELECT * FROM conversations", conn).to_dict('records'),
                        "alerts": pd.read_sql_query("SELECT * FROM alerts", conn).to_dict('records'),
                        "sessions": pd.read_sql_query("SELECT * FROM sessions", conn).to_dict('records'),
                        "user_details": pd.read_sql_query("SELECT * FROM user_details", conn).to_dict('records'),
                        "export_timestamp": datetime.now().isoformat()
                    }
                    
                    conn.close()
                    
                    json_data = json.dumps(data, indent=2, default=str)
                    st.download_button(
                        "📥 Download Complete Database",
                        data=json_data,
                        file_name=f"hotline_database_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
            except Exception as e:
                st.error(f"Export failed: {str(e)}")
    
    with col2:
        st.subheader("📈 Quick Stats")
        
        try:
            stats = st.session_state.db_manager.get_session_stats()
            
            st.metric("Database Size", f"{os.path.getsize(st.session_state.db_manager.db_path) / 1024:.1f} KB")
            st.metric("Active Sessions", stats["total_sessions"])
            st.metric("Pending Alerts", stats["pending_alerts"])
            
            # Show database file info
            st.info(f"Database: `{st.session_state.db_manager.db_path}`")
            
        except Exception as e:
            st.error(f"Error loading stats: {str(e)}")

def render_chat_interface():
    """Render the main chat interface"""
    
    # Chat Interface
    st.subheader("💬 Talk to Safety Agent")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="agent-message"><strong>Agent:</strong> {message["content"]}</div>', 
                           unsafe_allow_html=True)
    
    # User input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your message here...",
            placeholder="I'm here to listen. How are you feeling today?",
            key="user_input"
        )
    
    with col2:
        send_button = st.button("Send", type="primary", use_container_width=True)
    
    # Process message
    if (send_button or user_input) and user_input.strip():
        # Add user message
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input,
            "timestamp": datetime.now()
        })
        
        # Show processing
        with st.spinner("Agent is responding..."):
            try:
                # Get agent response
                response = asyncio.run(
                    st.session_state.agent_executor.process_message(user_input)
                )
                
                # Add agent response
                st.session_state.messages.append({
                    "role": "agent", 
                    "content": response,
                    "timestamp": datetime.now()
                })
                
                # Rerun to update chat
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing message: {str(e)}")
                st.exception(e)
    
    # Emergency buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚨 I need immediate help", type="secondary", use_container_width=True):
            emergency_msg = "I need immediate help. This is an emergency."
            st.session_state.messages.append({
                "role": "user", 
                "content": emergency_msg,
                "timestamp": datetime.now()
            })
            st.rerun()
    
    with col2:
        if st.button("💭 I'm having thoughts of self-harm", type="secondary", use_container_width=True):
            crisis_msg = "I'm having thoughts of self-harm and I'm scared."
            st.session_state.messages.append({
                "role": "user", 
                "content": crisis_msg,
                "timestamp": datetime.now()
            })
            st.rerun()
    
    with col3:
        if st.button("🔄 Clear Chat", type="secondary", use_container_width=True):
            clear_current_session()
            st.rerun()

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🆘 Discord Safety Hotline Agent</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="safety-alert">
    <h3>⚠️ This is a crisis support tool</h3>
    <p>If you're in immediate danger, please call 911 or your local emergency services.</p>
    <p>This agent can help connect you with resources and alert responders through Discord.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get configuration
    model_provider, model_name, api_key, temperature, discord_token, channel_id, current_tab = render_sidebar()
    
    # Route to different interfaces based on tab
    if current_tab == "📊 Dashboard":
        render_dashboard()
        return
    elif current_tab == "📋 Alerts":
        render_alerts_management()
        return
    elif current_tab == "💾 Data Export":
        render_data_export()
        return
    
    # Validate configuration for chat interface
    if not all([api_key, discord_token, channel_id]):
        st.warning("⚠️ Please configure all required settings in the sidebar to start.")
        return
    
    # Initialize agent
    if st.session_state.agent_executor is None:
        try:
            agent = SafetyHotlineAgent(
                model_provider=model_provider,
                model_name=model_name,
                api_key=api_key,
                discord_token=discord_token,
                channel_id=channel_id,
                db_manager=st.session_state.db_manager,
                temperature=temperature
            )
            st.session_state.agent_executor = agent
            st.success("✅ Safety Agent initialized successfully!")
        except Exception as e:
            st.error(f"❌ Failed to initialize agent: {str(e)}")
            return
    
    # Render chat interface
    render_chat_interface()

if __name__ == "__main__":
    main()input(
            "Discord Channel ID",
            help="Channel ID where alerts will be sent"
        )
    
    with st.sidebar.expander("📋 Emergency Resources", expanded=False):
        st.markdown("""
        **🚨 Emergency Numbers:**
        - 911 (US Emergency)
        - 988 (US Suicide Hotline)
        - 1-800-273-8255 (Crisis Line)
        
        **🌐 Online Resources:**
        - Crisis Text Line: Text HOME to 741741
        - National Suicide Prevention Lifeline
        - SAMHSA Helpline: 1-800-662-4357
        """)
    
    return model_provider, model_name, api_key, temperature, discord_token, channel_id

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">🆘 Discord Safety Hotline Agent</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="safety-alert">
    <h3>⚠️ This is a crisis support tool</h3>
    <p>If you're in immediate danger, please call 911 or your local emergency services.</p>
    <p>This agent can help connect you with resources and alert responders through Discord.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get configuration
    model_provider, model_name, api_key, temperature, discord_token, channel_id = render_sidebar()
    
    # Validate configuration
    if not all([api_key, discord_token, channel_id]):
        st.warning("⚠️ Please configure all required settings in the sidebar to start.")
        return
    
    # Initialize agent
    if st.session_state.agent_executor is None:
        try:
            agent = SafetyHotlineAgent(
                model_provider=model_provider,
                model_name=model_name,
                api_key=api_key,
                discord_token=discord_token,
                channel_id=channel_id,
                temperature=temperature
            )
            st.session_state.agent_executor = agent
            st.success("✅ Safety Agent initialized successfully!")
        except Exception as e:
            st.error(f"❌ Failed to initialize agent: {str(e)}")
            return
    
    # Chat Interface
    st.subheader("💬 Talk to Safety Agent")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message"><strong>You:</strong> {message["content"]}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="agent-message"><strong>Agent:</strong> {message["content"]}</div>', 
                           unsafe_allow_html=True)
    
    # User input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your message here...",
            placeholder="I'm here to listen. How are you feeling today?",
            key="user_input"
        )
    
    with col2:
        send_button = st.button("Send", type="primary", use_container_width=True)
    
    # Process message
    if (send_button or user_input) and user_input.strip():
        # Add user message
        st.session_state.messages.append({
            "role": "user", 
            "content": user_input,
            "timestamp": datetime.now()
        })
        
        # Show processing
        with st.spinner("Agent is responding..."):
            try:
                # Get agent response
                response = asyncio.run(
                    st.session_state.agent_executor.process_message(user_input)
                )
                
                # Add agent response
                st.session_state.messages.append({
                    "role": "agent", 
                    "content": response,
                    "timestamp": datetime.now()
                })
                
                # Rerun to update chat
                st.rerun()
                
            except Exception as e:
                st.error(f"Error processing message: {str(e)}")
                st.exception(e)
    
    # Emergency buttons
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚨 I need immediate help", type="secondary", use_container_width=True):
            emergency_msg = "I need immediate help. This is an emergency."
            st.session_state.messages.append({
                "role": "user", 
                "content": emergency_msg,
                "timestamp": datetime.now()
            })
            st.rerun()
    
    with col2:
        if st.button("💭 I'm having thoughts of self-harm", type="secondary", use_container_width=True):
            crisis_msg = "I'm having thoughts of self-harm and I'm scared."
            st.session_state.messages.append({
                "role": "user", 
                "content": crisis_msg,
                "timestamp": datetime.now()
            })
            st.rerun()
    
    with col3:
        if st.button("🔄 Clear Chat", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.session_state.safety_concern_detected = False
            st.session_state.collecting_details = False
            st.rerun()

if __name__ == "__main__":
    main()
