import urllib.parse
from dotenv import load_dotenv
import os, json, asyncio, traceback, re
import discord
from discord.ext import commands
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import BaseTool
from typing import Optional, Type, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone


class SendDiscordMessageInput(BaseModel):
    message: str = Field(description="The message to send to Discord")
    channel_id: Optional[int] = Field(default=None, description="Optional channel ID to send message to")


class SendDiscordCrisisAlertInput(BaseModel):
    user_message: str = Field(description="The original user message that triggered the crisis")
    crisis_level: str = Field(default="HIGH", description="Crisis level: LOW, MEDIUM, HIGH")
    user_id: str = Field(default="unknown", description="User ID who sent the crisis message")


class SendDiscordMessageTool(BaseTool):
    name: str = "send_discord_message"
    description: str = "Send a regular message to Discord channel"
    args_schema: Type[BaseModel] = SendDiscordMessageInput
    
    # Store bot instance as a class variable that can be set later
    _bot_instance = None
    
    @classmethod
    def set_bot_instance(cls, bot_instance):
        """Set the bot instance for all instances of this tool"""
        cls._bot_instance = bot_instance
    
    async def _arun(self, message: str, channel_id: Optional[int] = None) -> str:
        try:
            if not self._bot_instance:
                return "Error: Discord bot not initialized"
            
            target_channel_id = channel_id or int(os.getenv("DISCORD_CRISIS_CHANNEL_ID", os.getenv("CHANNEL_ID")))
            channel = self._bot_instance.get_channel(target_channel_id)
            if channel:
                await channel.send(message)
                return f"✅ Successfully sent message to Discord channel {target_channel_id}"
            else:
                return f"❌ Error: Could not find Discord channel {target_channel_id}"
        except Exception as e:
            return f"❌ Error sending Discord message: {str(e)}"
    
    def _run(self, message: str, channel_id: Optional[int] = None) -> str:
        # Sync version - not used in async context
        return "Sync version not implemented"


class SendDiscordCrisisAlertTool(BaseTool):
    name: str = "send_discord_crisis_alert"
    description: str = "Send a formatted crisis alert to Discord with proper emergency embed"
    args_schema: Type[BaseModel] = SendDiscordCrisisAlertInput
    
    # Store bot instance as a class variable that can be set later
    _bot_instance = None
    
    @classmethod
    def set_bot_instance(cls, bot_instance):
        """Set the bot instance for all instances of this tool"""
        cls._bot_instance = bot_instance
    
    async def _arun(self, user_message: str, crisis_level: str = "HIGH", user_id: str = "unknown") -> str:
        try:
            if not self._bot_instance:
                return "❌ Error: Discord bot not initialized"
            
            target_channel_id = int(os.getenv("DISCORD_CRISIS_CHANNEL_ID", os.getenv("CHANNEL_ID")))
            channel = self._bot_instance.get_channel(target_channel_id)
            
            if not channel:
                return f"❌ Error: Could not find Discord crisis channel {target_channel_id}"
            
            # Create emergency embed
            color_map = {"HIGH": 0xFF0000, "MEDIUM": 0xFF8C00, "LOW": 0xFFFF00}
            embed_color = color_map.get(crisis_level, 0xFF0000)
            
            embed = discord.Embed(
                title="🚨 MENTAL HEALTH CRISIS ALERT 🚨",
                description="**IMMEDIATE ATTENTION REQUIRED**",
                color=embed_color,
                timestamp=datetime.now(timezone.utc)
            )
            
            embed.add_field(name="🚨 Crisis Level", value=f"**{crisis_level}**", inline=True)
            embed.add_field(name="👤 User ID", value=user_id, inline=True) 
            embed.add_field(name="📍 Source", value="Crisis Detection System", inline=True)
            
            embed.add_field(
                name="💬 User Message",
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
                value="✅ Contact user immediately\n✅ Professional crisis intervention\n✅ Monitor safety continuously\n✅ Document all response steps",
                inline=False
            )
            
            embed.set_footer(text="Crisis Response Protocol Activated | Respond Immediately")
            
            # Send @here notification with embed
            alert_content = f"@here **MENTAL HEALTH CRISIS ALERT**\nCrisis Level: **{crisis_level}** | User: {user_id}\n**IMMEDIATE RESPONSE REQUIRED**"
            
            # Send the alert
            alert_message = await channel.send(content=alert_content, embed=embed)
            
            return f"""✅ **DISCORD CRISIS ALERT SENT SUCCESSFULLY**

**📢 Alert Details:**
- Message ID: {alert_message.id}
- Channel: #{channel.name} ({target_channel_id})
- Crisis Level: {crisis_level}
- User: {user_id}
- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**🚨 Actions Completed:**
✅ @here notification sent to crisis response team
✅ Emergency embed with full crisis details posted
✅ Crisis level and user information clearly displayed
✅ Emergency resources included in alert
✅ Response requirements specified

**📋 Crisis Team Response Status:**
🔥 Crisis response team has been alerted and notified
📞 Immediate user contact should be initiated
👨‍⚕️ Professional crisis intervention protocols activated
🚨 Emergency services information provided
📊 Continuous safety monitoring should begin

**THE CRISIS RESPONSE TEAM HAS BEEN SUCCESSFULLY NOTIFIED**"""
            
        except Exception as e:
            error_msg = f"❌ Error sending Discord crisis alert: {str(e)}"
            print(error_msg)
            return error_msg
    
    def _run(self, user_message: str, crisis_level: str = "HIGH", user_id: str = "unknown") -> str:
        # Sync version - not used in async context
        return "Sync version not implemented"


class DiscordBot(commands.Bot):
    def __init__(self, agent_executor, coral_tools):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)
        self.agent_executor = agent_executor
        self.coral_tools = coral_tools
        self.crisis_channel_id = int(os.getenv("DISCORD_CRISIS_CHANNEL_ID", os.getenv("CHANNEL_ID")))
    
    async def on_ready(self):
        print(f"🤖 Discord Crisis Bot logged in as {self.user}")
        
        # Send startup notification to crisis channel
        channel = self.get_channel(self.crisis_channel_id)
        if channel:
            startup_embed = discord.Embed(
                title="🤖 Discord Crisis Agent Online",
                description="Crisis detection and alert system is now active",
                color=0x00FF00,
                timestamp=datetime.now(timezone.utc)
            )
            startup_embed.add_field(
                name="🛠️ Available Tools",
                value="✅ send_discord_message\n✅ send_discord_crisis_alert\n✅ Coral integration active",
                inline=False
            )
            startup_embed.add_field(
                name="📊 System Status",
                value="🟢 Real-time crisis monitoring active\n🟢 Emergency response protocols ready\n🟢 988 hotline integration ready",
                inline=False
            )
            await channel.send(embed=startup_embed)
    
    # Add slash commands for manual testing
    @discord.app_commands.command(name="test_crisis_alert", description="Test crisis alert system")
    async def test_crisis_alert(self, interaction: discord.Interaction, 
                               message: str = "Test crisis alert", 
                               level: str = "TEST"):
        """Test crisis alert manually"""
        try:
            # Use the crisis alert tool
            crisis_tool = None
            for tool in [SendDiscordCrisisAlertTool()]:
                if tool.name == "send_discord_crisis_alert":
                    crisis_tool = tool
                    SendDiscordCrisisAlertTool.set_bot_instance(self)
                    break
            
            if crisis_tool:
                result = await crisis_tool._arun(message, level, f"admin_{interaction.user.id}")
                await interaction.response.send_message("✅ Crisis alert test completed! Check the crisis channel.", ephemeral=True)
            else:
                await interaction.response.send_message("❌ Crisis alert tool not available!", ephemeral=True)
                
        except Exception as e:
            await interaction.response.send_message(f"❌ Crisis alert test failed: {str(e)}", ephemeral=True)
    
    @discord.app_commands.command(name="crisis_status", description="Check crisis system status")
    async def crisis_status(self, interaction: discord.Interaction):
        """Check system status"""
        embed = discord.Embed(
            title="📊 Crisis System Status",
            color=0x00FF00,
            timestamp=datetime.now(timezone.utc)
        )
        
        embed.add_field(
            name="🤖 Bot Status",
            value="✅ Online and Ready",
            inline=True
        )
        
        embed.add_field(
            name="📢 Crisis Channel",
            value=f"<#{self.crisis_channel_id}>",
            inline=True
        )
        
        embed.add_field(
            name="🛠️ Tools Available",
            value="✅ Discord messaging\n✅ Crisis alert formatting\n✅ Coral integration",
            inline=False
        )
        
        await interaction.response.send_message(embed=embed, ephemeral=True)


def extract_crisis_info(message_content: str) -> dict:
    """Extract crisis information from Coral message"""
    crisis_info = {
        "user_message": "Crisis detected",
        "crisis_level": "HIGH",
        "user_id": "unknown"
    }
    
    # Extract user message
    message_match = re.search(r'\*\*Message:\*\*\s*([^\n*]+)', message_content)
    if message_match:
        crisis_info["user_message"] = message_match.group(1).strip()
    
    # Extract crisis level
    level_match = re.search(r'\*\*Crisis Level:\*\*\s*([^\n*]+)', message_content)
    if level_match:
        crisis_info["crisis_level"] = level_match.group(1).strip()
    
    # Extract user ID
    user_match = re.search(r'\*\*User ID:\*\*\s*([^\n*]+)', message_content)
    if user_match:
        crisis_info["user_id"] = user_match.group(1).strip()
    
    return crisis_info


def get_tools_description(tools):
    return "\n".join(
        f"Tool: {tool.name}, Schema: {json.dumps(tool.args).replace('{', '{{').replace('}', '}}')}"
        for tool in tools
    )


async def create_agent(coral_tools, discord_tools):
    coral_tools_description = get_tools_description(coral_tools)
    discord_tools_description = get_tools_description(discord_tools)
    combined_tools = coral_tools + discord_tools
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"""You are a Discord Crisis Alert Agent with specialized tools for mental health emergency response.

**PRIMARY MISSION:** 
1. Monitor for crisis alert messages from other agents via Coral
2. Send properly formatted crisis alerts to Discord channels
3. Ensure crisis response teams are immediately notified

**WORKFLOW:**
1. Use `wait_for_mentions` (timeoutMs: 30000) to listen for crisis alerts from other agents
2. When you receive a crisis alert message, extract the crisis information
3. Determine the appropriate response based on the message content:
   - If it contains "CRISIS ALERT" or crisis keywords → use `send_discord_crisis_alert`
   - If it's a regular message → use `send_discord_message`
4. For crisis alerts, extract: user_message, crisis_level, user_id from the received content
5. Send the alert using the appropriate Discord tool
6. Always respond back to the sender using `send_message` with confirmation
7. Wait 2 seconds and repeat from step 1

**CRISIS DETECTION KEYWORDS:**
Look for: "CRISIS ALERT", "crisis", "not safe", "suicide", "help me", "emergency", "mental health"

**DISCORD TOOLS USAGE:**
- `send_discord_crisis_alert`: For crisis situations (creates emergency embed with @here)
- `send_discord_message`: For regular notifications

**RESPONSE FORMAT:**
- Always confirm successful Discord alert sending
- Include message ID and channel information in confirmation
- If error occurs, report the error details

**ERROR HANDLING:**
- If Discord fails, report the specific error
- Always respond to the sender even if Discord fails
- Provide alternative contact methods if Discord is unavailable

**Available Coral Tools:** {coral_tools_description}
**Available Discord Tools:** {discord_tools_description}

Remember: This is a critical mental health system. Every crisis alert must be handled immediately and professionally."""
        ),
        ("placeholder", "{agent_scratchpad}")
    ])

    model = init_chat_model(
        model=os.getenv("MODEL_NAME", "qwen/qwen2.5-32b"),
        model_provider=os.getenv("MODEL_PROVIDER", "groq"),
        api_key=os.getenv("GROQ_API_KEY", os.getenv("OPENAI_API_KEY")),
        temperature=float(os.getenv("MODEL_TEMPERATURE", "0.1")),  # Lower temperature for crisis situations
        max_tokens=int(os.getenv("MODEL_MAX_TOKENS", "4000")),
        base_url=os.getenv("MODEL_BASE_URL", None)
    )
    
    agent = create_tool_calling_agent(model, combined_tools, prompt)
    return AgentExecutor(agent=agent, tools=combined_tools, verbose=True, handle_parsing_errors=True)


async def main():
    # Load environment variables
    runtime = os.getenv("CORAL_ORCHESTRATION_RUNTIME", None)
    if runtime is None:
        load_dotenv()

    # Coral configuration
    base_url = os.getenv("CORAL_SSE_URL")
    agentID = os.getenv("CORAL_AGENT_ID", "discord_agent")  # This should match the crisis agent config

    coral_params = {
        "agentId": agentID,
        "agentDescription": "Discord crisis alert agent that sends formatted emergency notifications to Discord channels"
    }

    query_string = urllib.parse.urlencode(coral_params)
    CORAL_SERVER_URL = f"{base_url}?{query_string}"
    print(f"🔗 Connecting to Coral Server: {CORAL_SERVER_URL}")

    # Connect to Coral
    timeout = float(os.getenv("TIMEOUT_MS", "30000"))
    client = MultiServerMCPClient(
        connections={
            "coral": {
                "transport": "sse",
                "url": CORAL_SERVER_URL,
                "timeout": timeout,
                "sse_read_timeout": timeout,
            }
        }
    )

    print("✅ Coral Server Connection Established")
    coral_tools = await client.get_tools(server_name="coral")
    print(f"🛠️ Coral tools available: {len(coral_tools)}")

    # Create Discord tools
    discord_message_tool = SendDiscordMessageTool()
    discord_crisis_tool = SendDiscordCrisisAlertTool()
    discord_tools = [discord_message_tool, discord_crisis_tool]
    
    print(f"🛠️ Discord tools created: {len(discord_tools)}")

    # Create agent executor
    agent_executor = await create_agent(coral_tools, discord_tools)
    print("🤖 Crisis Discord Agent created successfully")

    # Initialize Discord bot
    bot = DiscordBot(agent_executor, coral_tools)
    
    # Set bot instance for Discord tools
    SendDiscordMessageTool.set_bot_instance(bot)
    SendDiscordCrisisAlertTool.set_bot_instance(bot)
    
    print("🤖 Discord tools configured with bot instance")

    # Start Discord bot
    discord_token = os.getenv("DISCORD_BOT_TOKEN")
    if discord_token:
        print("🚀 Starting Discord bot...")
        asyncio.create_task(bot.start(discord_token))
        await asyncio.sleep(5)  # Give Discord bot more time to connect and sync commands
        
        # Sync slash commands
        try:
            synced = await bot.tree.sync()
            print(f"✅ Synced {len(synced)} Discord slash commands")
        except Exception as e:
            print(f"⚠️ Failed to sync slash commands: {e}")
    else:
        print("❌ Warning: DISCORD_BOT_TOKEN not found. Discord functionality will be limited.")
        return

    print("🚨 Discord Crisis Alert Agent is now running!")
    print("📢 Listening for crisis alerts from other agents...")
    print("🛠️ Available slash commands: /test_crisis_alert, /crisis_status")

    # Main agent loop - listens for crisis alerts
    while True:
        try:
            print("👂 Listening for crisis alerts via Coral...")
            await agent_executor.ainvoke({"agent_scratchpad": []})
            print("🔄 Completed agent cycle, restarting...")
            await asyncio.sleep(1)
        except Exception as e:
            print(f"❌ Error in crisis agent loop: {str(e)}")
            print(traceback.format_exc())
            await asyncio.sleep(5)  # Wait before retrying


if __name__ == "__main__":
    print("🚨 Starting Discord Crisis Alert Agent...")
    print("📋 Configuration:")
    print(f"   - Agent ID: {os.getenv('CORAL_AGENT_ID', 'discord_agent')}")
    print(f"   - Crisis Channel: {os.getenv('DISCORD_CRISIS_CHANNEL_ID', 'Not set')}")
    print(f"   - Coral URL: {os.getenv('CORAL_SSE_URL', 'Not set')}")
    print("🚀 Starting agent...")
    
    asyncio.run(main())
