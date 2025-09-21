import streamlit as st
import asyncio
import json
import re
import threading
import time
import logging
from typing import Dict, Any
from datetime import datetime, timezone

# LangGraph and LangChain imports
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent

# Discord imports
try:
    import discord
    from discord.ext import commands
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False
    st.error("❌ Discord.py not available. Install with: pip install discord.py")


# -------------------------------------------------------------------
# Crisis Detector
# -------------------------------------------------------------------
class CrisisDetector:
    """Advanced crisis detection with multiple keyword patterns"""

    CRISIS_KEYWORDS = [
        "suicide", "kill myself", "end my life", "want to die",
        "better off dead", "not safe", "harm myself", "hurt myself",
        "end it all", "give up", "no point living", "overdose",
        "can't go on", "hopeless", "worthless"
    ]

    CRISIS_PHRASES = [
        r"i.*want.*to.*die", r"i.*can't.*go.*on", r"i.*not.*safe",
        r"i.*hurt.*myself", r"end.*my.*life", r"better.*off.*dead",
        r"no.*point.*in.*living", r"going.*to.*kill", r"suicide.*plan"
    ]

    HOTLINES = {
        "US": {
            "primary": "988",
            "name": "988 Suicide & Crisis Lifeline",
            "text": "Text HOME to 741741 (Crisis Text Line)",
            "chat": "suicidepreventionlifeline.org",
            "emergency": "911"
        },
        "UK": {
            "primary": "116 123",
            "name": "Samaritans",
            "text": "Text SHOUT to 85258",
            "chat": "samaritans.org",
            "emergency": "999"
        },
        "CA": {
            "primary": "1-833-456-4566",
            "name": "Talk Suicide Canada",
            "text": "Text 45645",
            "chat": "talksuicide.ca",
            "emergency": "911"
        },
        "AU": {
            "primary": "13 11 14",
            "name": "Lifeline Australia",
            "text": "Text 0477 13 11 14",
            "chat": "lifeline.org.au",
            "emergency": "000"
        }
    }

    @staticmethod
    def detect_crisis(text: str) -> Dict[str, Any]:
        """Detect crisis indicators in text with severity levels"""
        text_lower = text.lower().strip()

        keywords_found = [kw for kw in CrisisDetector.CRISIS_KEYWORDS if kw in text_lower]
        phrases_found = [p for p in CrisisDetector.CRISIS_PHRASES if re.search(p, text_lower)]

        if keywords_found or phrases_found:
            high_risk = any(w in text_lower for w in ["suicide", "kill myself", "not safe", "end my life"])
            if high_risk:
                level = "HIGH"
            elif len(keywords_found) >= 2:
                level = "MEDIUM"
            else:
                level = "LOW"
        else:
            level = "NONE"

        return {
            "crisis_detected": level != "NONE",
            "crisis_level": level,
            "keywords_found": keywords_found,
            "phrases_found": phrases_found,
            "immediate_danger": level == "HIGH",
            "needs_emergency": "not safe" in text_lower or "kill myself" in text_lower
        }

    @staticmethod
    def get_hotline_response(country: str = "US") -> str:
        hotline = CrisisDetector.HOTLINES.get(country, CrisisDetector.HOTLINES["US"])
        return f"""🚨 **IMMEDIATE HELP AVAILABLE 24/7** 🚨

**{hotline['name']}**
📞 **CALL: {hotline['primary']}**
💬 **TEXT: {hotline['text']}**
🌐 **CHAT: {hotline['chat']}**

🚨 **Emergency: {hotline['emergency']}**
"""


# -------------------------------------------------------------------
# Discord Crisis Bot
# -------------------------------------------------------------------
class CrisisDiscordBot(commands.Bot):
    """Discord bot for crisis alerts"""

    def __init__(self, token: str, guild_id: int, crisis_channel_id: int):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!crisis_", intents=intents)

        self.token = token
        self.guild_id = guild_id
        self.crisis_channel_id = crisis_channel_id
        self.is_ready = False

    async def on_ready(self):
        logging.info(f"🚨 Crisis Bot logged in as {self.user}")
        self.is_ready = True


# Global instance
crisis_bot_instance = None


def start_discord_bot(token: str, guild_id: int, crisis_channel_id: int):
    """Start Discord bot in background safely"""
    global crisis_bot_instance

    if crisis_bot_instance and not crisis_bot_instance.is_closed():
        return crisis_bot_instance

    crisis_bot_instance = CrisisDiscordBot(token, guild_id, crisis_channel_id)

    def run_bot():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(crisis_bot_instance.start(token))

    threading.Thread(target=run_bot, daemon=True).start()
    time.sleep(3)
    return crisis_bot_instance


# -------------------------------------------------------------------
# Crisis Agent
# -------------------------------------------------------------------
class CrisisAgentWithTools:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.detector = CrisisDetector()
        self.llm = self._get_llm()

        self.discord_bot = None
        if DISCORD_AVAILABLE and config.get("discord_token") and config.get("discord_crisis_channel"):
            try:
                self.discord_bot = start_discord_bot(
                    token=config["discord_token"],
                    guild_id=int(config.get("discord_guild_id", 0)),
                    crisis_channel_id=int(config["discord_crisis_channel"])
                )
            except Exception as e:
                logging.error(f"❌ Discord bot init failed: {e}")

        self.tools = self.create_crisis_tools()
        self.agent = self.create_crisis_agent()

    def _get_llm(self):
        if self.config["provider"] == "openai":
            return ChatOpenAI(api_key=self.config["api_key"], model=self.config["model"], temperature=0.3)
        elif self.config["provider"] == "groq":
            return ChatGroq(api_key=self.config["api_key"], model=self.config["model"], temperature=0.3)

    def create_crisis_tools(self):
        @tool
        def detect_crisis_level(user_message: str) -> str:
            try:
                r = self.detector.detect_crisis(user_message)
                return f"Crisis Detected: {r['crisis_detected']}, Level: {r['crisis_level']}"
            except Exception as e:
                return f"❌ Detection error: {e}"

        @tool
        def provide_crisis_hotlines(country_code: str = "US") -> str:
            try:
                return self.detector.get_hotline_response(country_code)
            except Exception as e:
                return f"❌ Hotline error: {e}"

        return [detect_crisis_level, provide_crisis_hotlines]

    def create_crisis_agent(self):
        template = """You are a CRISIS SPECIALIST.

Tools:
{tools}

Tool names: {tool_names}

Question: {input}
Thought:{agent_scratchpad}"""

        prompt = PromptTemplate.from_template(template)
        agent = create_react_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True, return_intermediate_steps=True)

    def process_message(self, user_message: str, user_id: str = "streamlit_user") -> Dict[str, Any]:
        try:
            inp = {"input": f"User {user_id}: {user_message}"}
            resp = self.agent.invoke(inp)
            return {"output": resp["output"], "steps": resp.get("intermediate_steps", [])}
        except Exception as e:
            return {"error": str(e), "fallback": CrisisDetector.get_hotline_response("US")}


# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="🚨 Crisis Agent", page_icon="🚨", layout="wide")
    st.title("🚨 Mental Health Crisis Agent with Discord Alerts")

    with st.sidebar:
        st.header("⚙️ Configuration")
        provider = st.selectbox("LLM Provider", ["openai", "groq"])
        api_key = st.text_input("API Key", type="password")
        model = st.text_input("Model", "gpt-4o-mini")
        discord_token = st.text_input("Discord Token", type="password")
        discord_channel = st.text_input("Crisis Channel ID")

    user_input = st.text_area("Enter user message:")
    if st.button("Analyze"):
        config = {
            "provider": provider,
            "api_key": api_key,
            "model": model,
            "discord_token": discord_token,
            "discord_crisis_channel": discord_channel
        }
        agent = CrisisAgentWithTools(config)
        res = agent.process_message(user_input)
        st.write(res)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
