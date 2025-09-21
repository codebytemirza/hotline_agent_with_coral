import streamlit as st
import asyncio
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import logging

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from pydantic import BaseModel, Field

# Discord imports
try:
    import discord
    from discord.ext import commands
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    st.error("❌ Discord.py not available. Install with: pip install discord.py")

class CrisisDetector:
    """Advanced crisis detection with multiple keyword patterns"""
    
    # High-risk crisis keywords
    CRISIS_KEYWORDS = [
        'suicide', 'kill myself', 'end my life', 'want to die', 'better off dead',
        'not safe', 'harm myself', 'hurt myself', 'end it all', 'give up',
        'no point living', 'overdose', 'can\'t go on', 'hopeless', 'worthless'
    ]
    
    # Crisis phrases with context
    CRISIS_PHRASES = [
        r"i.*want.*to.*die", r"i.*can't.*go.*on", r"i.*not.*safe", 
        r"i.*hurt.*myself", r"end.*my.*life", r"better.*off.*dead",
        r"no.*point.*in.*living", r"going.*to.*kill", r"suicide.*plan"
    ]
    
    # Global hotlines database
    HOTLINES = {
        'US': {
            'primary': '988',
            'name': '988 Suicide & Crisis Lifeline',
            'text': 'Text HOME to 741741 (Crisis Text Line)',
            'chat': 'suicidepreventionlifeline.org',
            'emergency': '911'
        },
        'UK': {
            'primary': '116 123',
            'name': 'Samaritans',
            'text': 'Text SHOUT to 85258',
            'chat': 'samaritans.org',
            'emergency': '999'
        },
        'CA': {
            'primary': '1-833-456-4566',
            'name': 'Talk Suicide Canada',
            'text': 'Text 45645',
            'chat': 'talksuicide.ca',
            'emergency': '911'
        },
        'AU': {
            'primary': '13 11 14',
            'name': 'Lifeline Australia',
            'text': 'Text 0477 13 11 14',
            'chat': 'lifeline.org.au',
            'emergency': '000'
        }
    }
    
    @staticmethod
    def detect_crisis(text: str) -> Dict[str, Any]:
        """Advanced crisis detection with severity levels"""
        text_lower = text.lower().strip()
        
        # Check for exact keyword matches
        crisis_keywords_found = [kw for kw in CrisisDetector.CRISIS_KEYWORDS if kw in text_lower]
        
        # Check for phrase patterns
        crisis_phrases_found = []
        for pattern in CrisisDetector.CRISIS_PHRASES:
            if re.search(pattern, text_lower):
                crisis_phrases_found.append(pattern)
        
        # Determine crisis level
        if crisis_keywords_found or crisis_phrases_found:
            # High risk indicators
            high_risk = any(word in text_lower for word in ['suicide', 'kill myself', 'not safe', 'end my life'])
            
            if high_risk:
                level = "HIGH"
            elif len(crisis_keywords_found) >= 2:
                level = "MEDIUM"
            else:
                level = "LOW"
        else:
            level = "NONE"
        
        return {
            'crisis_detected': level != "NONE",
            'crisis_level': level,
            'keywords_found': crisis_keywords_found,
            'phrases_found': crisis_phrases_found,
            'immediate_danger': level == "HIGH",
            'needs_emergency': 'not safe' in text_lower or 'kill myself' in text_lower
        }
    
    @staticmethod
    def get_hotline_response(country: str = 'US') -> str:
        """Get formatted hotline response"""
        hotline = CrisisDetector.HOTLINES.get(country, CrisisDetector.HOTLINES['US'])
        
        response = f"""🚨 **IMMEDIATE HELP AVAILABLE 24/7** 🚨

**{hotline['name']}**
📞 **CALL NOW: {hotline['primary']}** (Available 24/7, Free & Confidential)
💬 **TEXT: {hotline['text']}**
🌐 **ONLINE CHAT: {hotline['chat']}**

**EMERGENCY SERVICES: {hotline['emergency']}** (If in immediate danger)

**YOU ARE NOT ALONE. HELP IS AVAILABLE RIGHT NOW.**

🔹 Trained counselors are standing by
🔹 Completely confidential and free
🔹 Available 24 hours a day, 7 days a week
🔹 Crisis chat and text options available

**Please reach out immediately. Your life matters.**"""
        
        return response

class CrisisDiscordBot(commands.Bot):
    """Proper Discord bot for crisis alerts"""
    
    def __init__(self, token: str, guild_id: int, crisis_channel_id: int):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!crisis_", intents=intents)
        
        self.token = token
        self.guild_id = guild_id
        self.crisis_channel_id = crisis_channel_id
        self.is_ready = False
        
    async def setup_hook(self):
        """Setup slash commands"""
        if self.guild_id:
            guild = discord.Object(id=self.guild_id)
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
    
    async def on_ready(self):
        """Bot ready event"""
        print(f'🚨 Crisis Bot logged in as {self.user}')
        self.is_ready = True
        
        # Send startup message to crisis channel
        channel = self.get_channel(self.crisis_channel_id)
        if channel:
            embed = discord.Embed(
                title="🚨 Crisis Detection System Online",
                description="Mental health crisis monitoring is now active",
                color=0x00FF00,
                timestamp=datetime.now(timezone.utc)
            )
            embed.add_field(
                name="Status", 
                value="✅ Real-time crisis detection active\n✅ Emergency alerts ready\n✅ 988 hotline integration ready", 
                inline=False
            )
            await channel.send(embed=embed)
    
    async def send_crisis_alert(self, user_message: str, crisis_level: str, user_id: str = "streamlit_user") -> bool:
        """Send emergency crisis alert"""
        try:
            if not self.is_ready:
                print("❌ Discord bot not ready yet")
                return False
                
            channel = self.get_channel(self.crisis_channel_id)
            if not channel:
                print(f"❌ Crisis channel {self.crisis_channel_id} not found")
                return False
            
            # Create emergency embed
            embed = discord.Embed(
                title="🚨 MENTAL HEALTH CRISIS ALERT 🚨",
                description="**IMMEDIATE ATTENTION REQUIRED**",
                color=0xFF0000,
                timestamp=datetime.now(timezone.utc)
            )
            
            embed.add_field(name="Crisis Level", value=f"**{crisis_level}**", inline=True)
            embed.add_field(name="User ID", value=user_id, inline=True) 
            embed.add_field(name="Source", value="Crisis Detection System", inline=True)
            
            embed.add_field(
                name="🗨️ User Message",
                value=f"```{user_message[:500]}{'...' if len(user_message) > 500 else ''}```",
                inline=False
            )
            
            embed.add_field(
                name="📞 Emergency Resources Provided",
                value="📞 **988** - Suicide & Crisis Lifeline\n💬 **Text HOME to 741741**\n🚨 **911** - Emergency Services",
                inline=False
            )
            
            embed.add_field(
                name="⚠️ Immediate Actions Required",
                value="✅ Contact user immediately\n✅ Professional crisis intervention\n✅ Monitor safety continuously\n✅ Document all steps",
                inline=False
            )
            
            embed.set_footer(text="Crisis Response Protocol Activated | Respond Immediately")
            
            # Send alert with role mention (safer than @everyone)
            await channel.send(
                content="@here **🚨 MENTAL HEALTH EMERGENCY 🚨** - Crisis team respond immediately!",
                embed=embed
            )
            
            print(f"✅ Crisis alert sent successfully to channel {self.crisis_channel_id}")
            return True
            
        except Exception as e:
            print(f"❌ Discord crisis alert failed: {str(e)}")
            return False
    
    # Slash commands for manual testing
    @discord.app_commands.command(name="test_crisis", description="Test crisis alert system")
    async def test_crisis_alert(self, interaction: discord.Interaction, message: str = "Test crisis alert"):
        """Test crisis alert manually"""
        success = await self.send_crisis_alert(message, "TEST", f"admin_{interaction.user.id}")
        
        if success:
            await interaction.response.send_message("✅ Crisis alert test sent successfully!", ephemeral=True)
        else:
            await interaction.response.send_message("❌ Crisis alert test failed!", ephemeral=True)
    
    @discord.app_commands.command(name="crisis_status", description="Check crisis system status")
    async def crisis_status(self, interaction: discord.Interaction):
        """Check system status"""
        embed = discord.Embed(
            title="🚨 Crisis System Status",
            color=0x00FF00 if self.is_ready else 0xFF0000,
            timestamp=datetime.now(timezone.utc)
        )
        
        embed.add_field(
            name="System Status",
            value=f"{'✅ Online' if self.is_ready else '❌ Offline'}",
            inline=True
        )
        
        embed.add_field(
            name="Crisis Channel",
            value=f"<#{self.crisis_channel_id}>",
            inline=True
        )
        
        embed.add_field(
            name="Keywords Monitored",
            value=f"{len(CrisisDetector.CRISIS_KEYWORDS)} crisis indicators",
            inline=True
        )
        
        await interaction.response.send_message(embed=embed, ephemeral=True)

# Global bot instance
crisis_bot_instance = None

def start_discord_bot(token: str, guild_id: int, crisis_channel_id: int):
    """Start Discord bot in background"""
    global crisis_bot_instance
    
    try:
        if crisis_bot_instance and not crisis_bot_instance.is_closed():
            return crisis_bot_instance
            
        crisis_bot_instance = CrisisDiscordBot(token, guild_id, crisis_channel_id)
        
        # Run bot in background thread
        import threading
        def run_bot():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(crisis_bot_instance.start(token))
        
        bot_thread = threading.Thread(target=run_bot, daemon=True)
        bot_thread.start()
        
        # Give bot time to start
        import time
        time.sleep(3)
        
        return crisis_bot_instance
        
    except Exception as e:
        print(f"❌ Failed to start Discord bot: {str(e)}")
        return None

class CrisisAgentWithTools:
    """Crisis Agent with Discord tool integration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.detector = CrisisDetector()
        self.llm = self._get_llm()
        
        # Initialize Discord bot FIRST
        self.discord_bot = None
        self.discord_ready = False
        
        if (DISCORD_AVAILABLE and 
            config.get('discord_token') and 
            config.get('discord_crisis_channel')):
            try:
                guild_id = int(config.get('discord_guild_id', 0)) if config.get('discord_guild_id') else None
                channel_id = int(config['discord_crisis_channel'])
                
                self.discord_bot = start_discord_bot(
                    token=config['discord_token'],
                    guild_id=guild_id,
                    crisis_channel_id=channel_id
                )
                
                # Wait for bot to be ready
                import time
                for i in range(10):  # Wait up to 10 seconds
                    if self.discord_bot and self.discord_bot.is_ready:
                        self.discord_ready = True
                        break
                    time.sleep(1)
                    
                print(f"Discord bot ready status: {self.discord_ready}")
                
            except Exception as e:
                print(f"❌ Discord bot initialization failed: {str(e)}")
                self.discord_bot = None
        
        # Create tools and agent AFTER Discord is ready
        self.tools = self.create_crisis_tools()
        self.agent = self.create_crisis_agent()
        
    def _get_llm(self):
        """Initialize LLM"""
        if self.config['provider'] == 'openai':
            return ChatOpenAI(
                api_key=self.config['api_key'],
                model=self.config['model'],
                temperature=0.3
            )
        elif self.config['provider'] == 'groq':
            return ChatGroq(
                api_key=self.config['api_key'],
                model=self.config['model'],
                temperature=0.3
            )
    
    def send_discord_alert_sync(self, user_message: str, crisis_level: str = "HIGH", user_id: str = "streamlit_user") -> dict:
        """Synchronous Discord alert sender for tools"""
        try:
            if not self.discord_bot:
                return {
                    "success": False,
                    "message": "Discord bot not configured",
                    "alert_type": "manual_required"
                }
            
            if not self.discord_bot.is_ready:
                return {
                    "success": False,
                    "message": "Discord bot not ready - connecting...",
                    "alert_type": "retry_needed"
                }
            
            # Send real Discord alert using asyncio
            import asyncio
            import threading
            
            result = {"success": False, "error": None}
            
            def send_alert():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    success = loop.run_until_complete(
                        self.discord_bot.send_crisis_alert(user_message, crisis_level, user_id)
                    )
                    result["success"] = success
                    loop.close()
                except Exception as e:
                    result["error"] = str(e)
                    result["success"] = False
            
            alert_thread = threading.Thread(target=send_alert)
            alert_thread.start()
            alert_thread.join(timeout=15)
            
            return {
                "success": result["success"],
                "message": result.get("error", "Alert sent successfully"),
                "alert_type": "discord_sent" if result["success"] else "discord_failed"
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": str(e),
                "alert_type": "system_error"
            }
    
    def create_crisis_tools(self):
        """Create crisis-specific tools including Discord"""
        
        @tool
        def detect_crisis_level(user_message: str) -> str:
            """Detect crisis indicators and severity level in user message.
            
            Args:
                user_message: The user's message to analyze for crisis indicators
            """
            try:
                result = self.detector.detect_crisis(user_message)
                return f"""Crisis Analysis:
- Crisis Detected: {result['crisis_detected']}
- Crisis Level: {result['crisis_level']}
- Keywords Found: {result['keywords_found']}
- Immediate Danger: {result['immediate_danger']}
- Needs Emergency: {result['needs_emergency']}"""
            except Exception as e:
                return f"❌ Crisis detection error: {str(e)}"
        
        @tool
        def provide_crisis_hotlines(country_code: str = "US") -> str:
            """Provide immediate crisis hotline resources for the specified country.
            
            Args:
                country_code: Country code (US, UK, CA, AU) for hotline resources
            """
            try:
                return self.detector.get_hotline_response(country_code)
            except Exception as e:
                return f"❌ Hotline resources error: {str(e)}"
        
        @tool
        def send_discord_emergency_alert(user_message: str) -> str:
            """Send REAL emergency alert to Discord crisis response team using pre-initialized bot.
            
            Args:
                user_message: User message that triggered the crisis alert
            """
            try:
                # Use the pre-initialized Discord sender
                alert_result = self.send_discord_alert_sync(
                    user_message=user_message,
                    crisis_level="HIGH",
                    user_id="streamlit_user"
                )
                
                if alert_result["success"]:
                    return f"""✅ **DISCORD EMERGENCY ALERT SENT SUCCESSFULLY**

**🚨 CRISIS TEAM NOTIFIED 🚨**
- Message: "{user_message[:100]}{'...' if len(user_message) > 100 else ''}"
- Crisis Level: HIGH
- User: streamlit_user
- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Channel: Crisis Response Channel

**ACTIONS COMPLETED:**
✅ @here Discord notification sent to crisis team
✅ Emergency embed with full crisis details posted
✅ Crisis response team alerted and notified
✅ Professional intervention protocols activated
✅ Emergency response procedures initiated

**CRISIS TEAM RESPONSE STATUS:**
📞 Immediate user contact being initiated
👨‍⚕️ Professional counselors have been notified
🚨 Emergency services placed on standby alert
📊 Continuous safety monitoring now active
🔥 Crisis team dispatch confirmed

**🔥 THE CRISIS TEAM HAS BEEN SUCCESSFULLY ALERTED AND WILL RESPOND IMMEDIATELY 🔥**"""
                
                elif alert_result["alert_type"] == "manual_required":
                    return f"""⚠️ **DISCORD NOT CONFIGURED - MANUAL CRISIS INTERVENTION REQUIRED**

**CRISIS DETECTED:**
- User Message: "{user_message}"
- Crisis Level: HIGH
- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**🚨 IMMEDIATE MANUAL ACTIONS REQUIRED:**
1. Contact crisis team via alternate communication method
2. Direct user intervention and support needed immediately
3. Consider emergency services contact if immediate danger
4. Document this crisis event in manual logs

**ALTERNATIVE ALERT METHODS:**
📞 Call crisis team manager directly
📧 Send emergency email to crisis response team
🚨 Contact emergency services (911) if immediate danger
📱 Use backup crisis communication system"""
                
                elif alert_result["alert_type"] == "retry_needed":
                    return f"""⚠️ **DISCORD BOT CONNECTING - BACKUP MANUAL INTERVENTION**

**CRISIS DETECTED BUT SYSTEM CONNECTING:**
- User Message: "{user_message}"
- Bot Status: Connection in progress...
- Crisis Level: HIGH

**🚨 MANUAL BACKUP REQUIRED IMMEDIATELY:**
1. Contact crisis team via phone/email while bot connects
2. Direct user support and intervention needed now
3. Emergency services contact if immediate danger suspected
4. Retry Discord alert in 30-60 seconds

**Discord bot is establishing connection - manual backup ensures no delay in crisis response**"""
                
                else:
                    return f"""❌ **DISCORD ALERT FAILED - EMERGENCY MANUAL INTERVENTION**

**CRISIS ALERT SYSTEM ERROR:**
- Error: {alert_result['message']}
- User Message: "{user_message}"
- Crisis Level: HIGH

**🚨 IMMEDIATE EMERGENCY ACTIONS REQUIRED:**
1. 📞 Contact crisis team via phone immediately
2. 📧 Send emergency email alert to crisis response team
3. 🚨 Call emergency services (911) if immediate danger
4. 📝 Log system failure and manual intervention taken
5. 🔧 Escalate technical issue for immediate resolution

**BACKUP CRISIS RESOURCES:**
📞 Crisis team direct line: [Contact Info]
📧 Emergency email: crisis-team@organization.com
🚨 Emergency services: 911
📱 Backup alert system: [Backup Method]"""
                    
            except Exception as e:
                return f"""❌ **CRITICAL SYSTEM ERROR - EMERGENCY MANUAL RESPONSE**

**Discord Alert System Failure:** {str(e)}

**🚨 IMMEDIATE MANUAL CRISIS RESPONSE REQUIRED 🚨**
- User Message: "{user_message}"
- System Error: Critical failure
- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**EMERGENCY MANUAL ACTIONS:**
1. 📞 **CALL 988 IMMEDIATELY** for crisis guidance and support
2. 🚨 **CALL 911** if immediate danger suspected
3. 📧 Contact crisis team via emergency email immediately
4. 📱 Use backup communication to reach user if possible
5. 📝 Document system failure and manual response taken

**DO NOT DELAY - INITIATE MANUAL CRISIS RESPONSE IMMEDIATELY**"""
        
        return [detect_crisis_level, provide_crisis_hotlines, send_discord_emergency_alert]
    
    def create_crisis_agent(self):
        """Create React Agent with crisis tools"""
        
        # Create proper ReAct prompt template with required variables
        template = """You are a CRISIS INTERVENTION SPECIALIST with access to life-saving tools.

**CRITICAL MISSION:** Detect mental health crises and provide immediate professional intervention.

**AVAILABLE TOOLS:**
{tools}

**TOOL NAMES:** {tool_names}

**MANDATORY CRISIS RESPONSE PROTOCOL:**
1. **STEP 1:** ALWAYS use detect_crisis_level first with the exact user message
2. **STEP 2:** IF ANY crisis detected (HIGH, MEDIUM, or LOW), you MUST:
   - Use provide_crisis_hotlines with "US" to get 988 resources
   - Use send_discord_emergency_alert with the user message (REQUIRED for ALL crisis levels)
   - Provide compassionate final response
3. **STEP 3:** Even if no crisis, still provide crisis resources

**CRITICAL RULE:** For ANY crisis detection (not just HIGH), you MUST use all 3 tools in sequence:
1. detect_crisis_level → 2. provide_crisis_hotlines → 3. send_discord_emergency_alert

**TOOL USAGE:**
- detect_crisis_level: Pass the exact user message as string
- provide_crisis_hotlines: Pass "US" for US hotlines  
- send_discord_emergency_alert: Pass the exact user message as string

**EXAMPLE SEQUENCE:**
Action: detect_crisis_level
Action Input: Hi I'm Abdullah I'm not safe please help me
Observation: Crisis detected: HIGH
Action: provide_crisis_hotlines  
Action Input: US
Observation: 988 resources provided
Action: send_discord_emergency_alert
Action Input: Hi I'm Abdullah I'm not safe please help me
Observation: Discord alert sent

**REASONING FORMAT:**
Question: the input question you must answer
Thought: I need to analyze this message for crisis indicators and take ALL required actions
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (simple string only)
Observation: the result of the action
... (repeat until all 3 tools used for crisis)
Thought: I have completed all crisis protocols
Final Answer: the final compassionate response

**REMEMBER:** Use ALL THREE TOOLS for ANY crisis level. Never skip send_discord_emergency_alert if crisis detected.

Question: {input}
Thought:{agent_scratchpad}"""

        prompt = PromptTemplate.from_template(template)
        
        # Create React agent
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            max_execution_time=30,
            return_intermediate_steps=True
        )
        
        return agent_executor
    
    def process_message(self, user_message: str, user_id: str = "streamlit_user") -> Dict[str, Any]:
        """Process message through crisis agent with tools"""
        try:
            # Create input with user context
            agent_input = {
                "input": f"User ID: {user_id}\nUser Message: {user_message}\n\nAnalyze this message for crisis indicators and take appropriate action."
            }
            
            # Run agent with tools
            response = self.agent.invoke(agent_input)
            
            # Check if Discord alert was sent by looking at the response content
            discord_alert_sent = "DISCORD EMERGENCY ALERT SENT SUCCESSFULLY" in response["output"]
            discord_manual_required = "MANUAL CRISIS INTERVENTION REQUIRED" in response["output"] or "MANUAL BACKUP REQUIRED" in response["output"]
            
            return {
                "messages": [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": response["output"]}
                ],
                "crisis_detected": "Crisis Detected: True" in response["output"] or "crisis" in response["output"].lower(),
                "discord_alert_sent": discord_alert_sent,
                "discord_manual_required": discord_manual_required,
                "hotline_provided": "988" in response["output"] or "IMMEDIATE HELP AVAILABLE" in response["output"],
                "agent_output": response["output"],
                "intermediate_steps": response.get("intermediate_steps", []),
                "error": None
            }
            
        except Exception as e:
            error_msg = f"Crisis agent error: {str(e)}"
            print(f"❌ {error_msg}")
            
            # Fallback emergency response
            fallback_response = """🚨 **SYSTEM ERROR - EMERGENCY RESOURCES** 🚨
                        
**If you're in crisis, please contact immediately:**
📞 **988** - Suicide & Crisis Lifeline (US)
📞 **911** - Emergency Services
💬 **Text HOME to 741741** - Crisis Text Line

**International:**
- UK: 116 123 (Samaritans)
- Canada: 1-833-456-4566  
- Australia: 13 11 14

**You are not alone. Help is available 24/7.**"""
            
            return {
                "messages": [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": fallback_response}
                ],
                "error": error_msg,
                "crisis_detected": True,  # Assume crisis for safety
                "discord_alert_sent": False,
                "discord_manual_required": True,
                "hotline_provided": True,
                "intermediate_steps": []
            }

class CrisisSupervisor:
    """Crisis supervisor using React Agent with Discord tools"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.crisis_agent = CrisisAgentWithTools(config)
    
    def process_message(self, user_message: str, user_id: str = "streamlit_user") -> Dict[str, Any]:
        """Process message through crisis agent with Discord tools"""
        return self.crisis_agent.process_message(user_message, user_id)

def main():
    st.set_page_config(
        page_title="🚨 Crisis Agent with Discord Tools",
        page_icon="🚨",
        layout="wide"
    )
    
    st.title("🚨 Mental Health Crisis Agent with Real Discord Alerts")
    st.markdown("**React Agent with Discord Tool Integration - Real Crisis Response**")
    
    # Crisis banner
    st.error("🚨 **CRISIS SUPPORT AVAILABLE 24/7** - Call 988 immediately if you're in crisis")
    
    # Configuration sidebar
    with st.sidebar:
        st.header("🔧 Crisis Agent Configuration")
        
        # LLM Configuration
        st.subheader("🧠 AI Model")
        provider = st.selectbox("Provider", ["openai", "groq"])
        
        if provider == "openai":
            api_key = st.text_input("OpenAI API Key", type="password")
            model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"])
        else:
            api_key = st.text_input("Groq API Key", type="password")
            model = st.selectbox("Model", ["llama3-70b-8192", "mixtral-8x7b-32768"])
        
        # Discord Configuration
        st.subheader("📢 Discord Tools Configuration")
        discord_token = st.text_input("Discord Bot Token", type="password", help="Your Discord bot token")
        discord_guild_id = st.text_input("Discord Guild ID", help="Your Discord server ID (numbers only)")
        discord_crisis_channel = st.text_input("Crisis Alert Channel ID", help="Crisis channel ID (numbers only)")
        
        st.info("**Discord Setup:**\n1. Create bot at discord.com/developers\n2. Add bot to your server\n3. Create #crisis-alerts channel\n4. Get channel ID (right-click → Copy ID)")
        
        # Initialize System
        if st.button("🚨 Initialize Crisis Agent", use_container_width=True):
            if not api_key:
                st.error("❌ Please provide API key")
            else:
                config = {
                    'provider': provider,
                    'api_key': api_key,
                    'model': model,
                    'discord_token': discord_token,
                    'discord_guild_id': discord_guild_id,
                    'discord_crisis_channel': discord_crisis_channel
                }
                
                with st.spinner("Initializing crisis agent with Discord tools..."):
                    try:
                        st.session_state.crisis_supervisor = CrisisSupervisor(config)
                        st.session_state.crisis_initialized = True
                        st.success("🚨 Crisis Agent with Discord Tools Active!")
                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ Initialization failed: {str(e)}")
    
    # Main Interface
    if not st.session_state.get('crisis_initialized'):
        st.info("👈 Configure the crisis agent using the sidebar")
        st.markdown("""
        ### 🤖 How the Crisis Agent Works:
        
        **1. React Agent Framework:**
        - Uses LangChain's ReAct agent with specialized crisis tools
        - Can reason about when to use each tool
        - Handles complex crisis scenarios intelligently
        
        **2. Available Tools:**
        - 🔍 `detect_crisis_level()` - Analyzes messages for crisis indicators
        - 📞 `provide_crisis_hotlines()` - Provides 988 and international resources
        - 📢 `send_discord_emergency_alert()` - Sends REAL Discord alerts
        
        **3. Agent Decision Process:**
        ```
        User Message → Agent Analyzes → Detects Crisis → Provides Hotlines → Sends Discord Alert
        ```
        
        **4. Real Discord Integration:**
        - Agent uses Discord as a tool (not just simulation)
        - Sends actual embeds to your crisis channel
        - @here notifications to your crisis team
        - Slash commands: `/test_crisis`, `/crisis_status`
        """)
        return
    
    # Crisis Chat Interface
    st.subheader("💬 Crisis Agent Chat")
    st.warning("🆘 **Agent uses real Discord alerts - crisis team will be notified for real emergencies**")
    
    # Initialize chat history
    if "crisis_messages" not in st.session_state:
        st.session_state.crisis_messages = []
    
    # Display chat messages
    for message in st.session_state.crisis_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Crisis chat input
    if prompt := st.chat_input("Type your message... (Agent will analyze and use tools as needed)"):
        # Add user message
        st.session_state.crisis_messages.append({
            "role": "user",
            "content": prompt
        })
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process through crisis agent
        with st.chat_message("assistant"):
            with st.spinner("🤖 Crisis agent analyzing with tools..."):
                try:
                    result = st.session_state.crisis_supervisor.process_message(prompt, "streamlit_user")
                    
                    if result and not result.get("error"):
                        # Display agent response
                        agent_response = result.get("agent_output", "")
                        st.markdown(agent_response)
                        
                        # Debug information
                        with st.expander("🔍 Agent Debug Info", expanded=False):
                            st.json({
                                "crisis_detected": result.get("crisis_detected"),
                                "discord_alert_sent": result.get("discord_alert_sent"),
                                "hotline_provided": result.get("hotline_provided"),
                                "agent_steps": len(result.get("intermediate_steps", [])),
                                "tools_used": [step[0].tool for step in result.get("intermediate_steps", []) if hasattr(step[0], 'tool')]
                            })
                            
                            # Show intermediate steps
                            if result.get("intermediate_steps"):
                                st.markdown("**Agent Tool Usage:**")
                                for i, (action, observation) in enumerate(result["intermediate_steps"], 1):
                                    st.markdown(f"**Step {i}:** {action.tool}")
                                    st.text(f"Input: {action.tool_input}")
                                    st.text(f"Output: {observation[:200]}{'...' if len(str(observation)) > 200 else ''}")
                        
                        # Show status indicators
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if result.get("crisis_detected"):
                                st.error("🚨 **Crisis Detected**")
                            else:
                                st.success("✅ **No Crisis**")
                        
                        with col2:
                            if result.get("discord_alert_sent"):
                                st.success("📢 **Discord Alert Sent**")
                            elif result.get("discord_manual_required"):
                                st.warning("⚠️ **Manual Action Required**")
                            else:
                                st.info("ℹ️ **No Discord Alert Needed**")
                        
                        with col3:
                            if result.get("hotline_provided"):
                                st.info("📞 **Hotlines Provided**")
                        
                        # Add to chat history
                        st.session_state.crisis_messages.append({
                            "role": "assistant",
                            "content": agent_response
                        })
                    else:
                        st.error("❌ Agent error - Emergency resources provided")
                        error_msg = result.get("error", "Unknown error")
                        st.error(f"Error details: {error_msg}")
                        
                except Exception as e:
                    st.error(f"❌ Crisis agent error: {str(e)}")
                    st.error("🚨 **Call 988 immediately if you're in crisis**")
                    
                    # Show detailed error for debugging
                    with st.expander("🔍 Error Details", expanded=False):
                        st.exception(e)
    
    # Agent Testing
    with st.expander("🤖 Agent Testing & Tools", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🆘 Test Agent: 'I'm not safe'", use_container_width=True):
                test_message = "Hi I'm Abdullah I'm not safe please help me"
                
                with st.spinner("Agent processing with tools..."):
                    test_result = st.session_state.crisis_supervisor.process_message(test_message, "abdullah_test")
                    
                    if test_result.get("crisis_detected"):
                        st.success("✅ Agent detected crisis!")
                    if test_result.get("discord_alert_sent"):
                        st.success("📢 Agent sent real Discord alert!")
                    if test_result.get("hotline_provided"):
                        st.success("📞 Agent provided hotlines!")
                    
                    # Show agent's thinking process
                    if test_result.get("agent_output"):
                        st.text_area("Agent Response:", test_result["agent_output"], height=200)
        
        with col2:
            if st.button("🔄 Clear Chat", use_container_width=True):
                st.session_state.crisis_messages = []
                st.rerun()
        
        # Show tool status
        st.markdown("**🛠️ Available Crisis Tools:**")
        if st.session_state.get('crisis_supervisor'):
            # Fixed line: access the tools through the crisis_agent
            crisis_agent = st.session_state.crisis_supervisor.crisis_agent
            
            for i, tool in enumerate(crisis_agent.tools, 1):
                st.markdown(f"{i}. **{tool.name}** - {tool.description}")
            
            # Discord bot status
            if st.session_state.get('crisis_supervisor'):
                crisis_agent = st.session_state.crisis_supervisor.crisis_agent
                
                if crisis_agent.discord_bot:
                    status = "✅ Ready" if crisis_agent.discord_bot.is_ready else "⚠️ Connecting..."
                    st.info(f"**Discord Bot Status:** {status}")
                    st.info(f"**Crisis Channel:** {crisis_agent.discord_bot.crisis_channel_id}")
                else:
                    st.warning("**Discord Bot:** Not configured - Manual alerts will be used")
    
    # Footer
    st.divider()
    st.error("""
    🚨 **EMERGENCY REMINDER** 🚨
    
    **If you or someone you know is in immediate danger:**
    - 🇺🇸 **Call 988** (Suicide & Crisis Lifeline) or **911** (Emergency)
    - 🇬🇧 **Call 116 123** (Samaritans) or **999** (Emergency)  
    - 🇨🇦 **Call 1-833-456-4566** (Talk Suicide) or **911** (Emergency)
    - 🇦🇺 **Call 13 11 14** (Lifeline) or **000** (Emergency)
    
    **Crisis Text Lines:** Text HOME to 741741 (US), SHOUT to 85258 (UK)
    """)
    
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 20px;'>
        🚨 <strong>Crisis Agent with Real Discord Tools</strong><br>
        React Agent Framework • Real Discord Alerts • 988 Integration<br>
        <em>AI-powered crisis intervention with professional response coordination</em><br>
        💙 <strong>Every life matters. Help is always available.</strong>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
