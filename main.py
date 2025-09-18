import urllib.parse
from dotenv import load_dotenv
import os
import json
import asyncio
import logging
import threading
import queue
import sys
from typing import List, Dict, Any, Optional
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_tool_calling_agent, AgentExecutor

# Available Coral tools
LIST_AGENTS_TOOL = "list_agents"
CREATE_THREAD_TOOL = "create_thread"
SEND_MESSAGE_TOOL = "send_message"
WAIT_FOR_MENTIONS_TOOL = "wait_for_mentions"
ADD_PARTICIPANT_TOOL = "add_participant"
CLOSE_THREAD_TOOL = "close_thread"

MAX_CHAT_HISTORY = 3
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 8000
SLEEP_INTERVAL = 1
ERROR_RETRY_INTERVAL = 5
WAIT_TIMEOUT_MS = 30000  # 30 seconds

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class UserInputHandler:
    def __init__(self):
        self.input_queue = queue.Queue()
        self.running = True
    
    def start_input_thread(self):
        """Start a separate thread to handle user input"""
        def input_worker():
            print("\n" + "="*60)
            print("CRISIS SUPPORT TRIAGE AGENT - USER INPUT INTERFACE")
            print("="*60)
            print("You can now simulate caller interactions with the triage agent.")
            print("Type messages as if you are someone calling the crisis support line.")
            print("The agent will assess your situation and connect you with counselors.")
            print("Type 'quit' or 'exit' to stop the program.")
            print("="*60 + "\n")
            
            while self.running:
                try:
                    user_input = input("Caller: ").strip()
                    if user_input.lower() in ['quit', 'exit', 'stop']:
                        print("Stopping the triage agent...")
                        self.running = False
                        self.input_queue.put(None)
                        break
                    elif user_input:
                        self.input_queue.put(user_input)
                        print(f"Message received: '{user_input}'")
                    else:
                        print("Please enter a message or 'quit' to exit.")
                except EOFError:
                    print("\nInput ended, stopping agent...")
                    self.running = False
                    self.input_queue.put(None)
                    break
                except KeyboardInterrupt:
                    print("\nInterrupted by user, stopping agent...")
                    self.running = False
                    self.input_queue.put(None)
                    break
        
        thread = threading.Thread(target=input_worker, daemon=True)
        thread.start()
        return thread
    
    def get_input(self):
        """Get user input from queue (non-blocking)"""
        try:
            return self.input_queue.get_nowait()
        except queue.Empty:
            return None
    
    def stop(self):
        """Stop the input handler"""
        self.running = False

def load_config() -> Dict[str, Any]:
    runtime = os.getenv("CORAL_ORCHESTRATION_RUNTIME", None)
    if runtime is None:
        load_dotenv()
    
    config = {
        "runtime": os.getenv("CORAL_ORCHESTRATION_RUNTIME", None),
        "coral_sse_url": os.getenv("CORAL_SSE_URL"),
        "agent_id": os.getenv("CORAL_AGENT_ID", "triage_interface_agent"),
        "model_name": os.getenv("MODEL_NAME"),
        "model_provider": os.getenv("MODEL_PROVIDER"),
        "api_key": os.getenv("GROQ_API_KEY"),
        "model_temperature": float(os.getenv("MODEL_TEMPERATURE", DEFAULT_TEMPERATURE)),
        "model_token": int(os.getenv("MODEL_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
        "base_url": os.getenv("BASE_URL")
    }
    
    required_fields = ["coral_sse_url", "agent_id", "model_name", "model_provider", "api_key"]
    missing = [field for field in required_fields if not config[field]]
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    
    if not 0 <= config["model_temperature"] <= 2:
        raise ValueError(f"Model temperature must be between 0 and 2, got {config['model_temperature']}")
    
    if config["model_token"] <= 0:
        raise ValueError(f"Model token must be positive, got {config['model_token']}")
    
    return config

def get_tools_description(tools: List[Any]) -> str:
    descriptions = []
    for tool in tools:
        tool_desc = f"Tool: {tool.name}, Schema: {json.dumps(tool.args).replace('{', '{{').replace('}', '}}')}"
        descriptions.append(tool_desc)
    
    return "\n".join(descriptions)

def format_chat_history(chat_history: List[Dict[str, str]]) -> str:
    if not chat_history:
        return "No previous chat history available."
    
    history_str = "Previous Conversations (use this to understand caller context and risk assessment):\n"
    
    for i, chat in enumerate(chat_history, 1):
        history_str += f"Conversation {i}:\n"
        history_str += f"Caller: {chat['user_input']}\n"
        history_str += f"Triage Agent: {chat['response']}\n\n"
    
    return history_str

async def wait_for_user_message(agent_tools: Dict[str, Any], user_input_handler: UserInputHandler, thread_id: Optional[str] = None) -> str:
    """Wait for a message from terminal user input or coral messaging"""
    
    # First check for terminal user input (higher priority)
    user_input = user_input_handler.get_input()
    if user_input is not None:
        if user_input == "":  # Empty string means quit signal
            return None
        logger.info(f"Received terminal input: {user_input}")
        return user_input
    
    # If no terminal input, check for coral messages
    try:
        result = await agent_tools[WAIT_FOR_MENTIONS_TOOL].ainvoke({
            "timeoutMs": 5000  # Shorter timeout when checking coral messages
        })
        
        # Parse the result to extract message
        if isinstance(result, dict):
            messages = result.get('messages', [])
            if messages:
                latest_message = messages[-1]
                coral_input = latest_message.get('content', 'No input provided')
                logger.info(f"Received coral message: {coral_input}")
                return coral_input
        elif isinstance(result, str):
            logger.info(f"Received coral message: {result}")
            return result
        
        return "No message received"
        
    except Exception as e:
        # Timeout is expected when no coral messages
        if "timeout" not in str(e).lower():
            logger.error(f"Error waiting for coral message: {str(e)}")
        return "No message received"

async def send_triage_response(agent_tools: Dict[str, Any], response: str, thread_id: Optional[str] = None, mentions: List[str] = None) -> None:
    """Send response via thread message or terminal output"""
    print(f"Triage Agent: {response}")
    
    try:
        if thread_id:
            # Send message to existing coral thread
            await agent_tools[SEND_MESSAGE_TOOL].ainvoke({
                "threadId": thread_id,
                "content": response,
                "mentions": mentions or []
            })
            logger.info(f"Sent response to coral thread: {thread_id}")
        else:
            # Response displayed in terminal, also log it
            logger.info(f"Terminal response: {response}")
            
    except Exception as e:
        logger.error(f"Error sending response: {str(e)}")

async def find_available_counselors(agent_tools: Dict[str, Any]) -> List[str]:
    """Find available counselor agents"""
    try:
        result = await agent_tools[LIST_AGENTS_TOOL].ainvoke({
            "includeDetails": True
        })
        
        # Parse result to find counselor agents
        counselor_ids = []
        if isinstance(result, dict):
            agents = result.get('agents', [])
            for agent in agents:
                agent_id = agent.get('id', '')
                description = agent.get('description', '').lower()
                # Look for counselor-related keywords
                if any(keyword in description for keyword in ['counselor', 'therapist', 'crisis', 'support']):
                    counselor_ids.append(agent_id)
        
        logger.info(f"Found {len(counselor_ids)} potential counselors: {counselor_ids}")
        return counselor_ids
        
    except Exception as e:
        logger.error(f"Error finding counselors: {str(e)}")
        return []

async def create_counselor_thread(agent_tools: Dict[str, Any], caller_context: str) -> Optional[str]:
    """Create a thread with available counselors"""
    try:
        counselor_ids = await find_available_counselors(agent_tools)
        
        if not counselor_ids:
            logger.warning("No counselors available")
            return None
        
        # Create thread with counselors
        result = await agent_tools[CREATE_THREAD_TOOL].ainvoke({
            "threadName": f"Crisis Support - {asyncio.get_event_loop().time()}",
            "participantIds": counselor_ids
        })
        
        if isinstance(result, dict):
            thread_id = result.get('threadId')
            if thread_id:
                # Send initial context to counselors
                context_message = f"HANDOFF FROM TRIAGE: {caller_context}"
                await agent_tools[SEND_MESSAGE_TOOL].ainvoke({
                    "threadId": thread_id,
                    "content": context_message,
                    "mentions": counselor_ids
                })
                logger.info(f"Created counselor thread: {thread_id}")
                return thread_id
        
        return None
        
    except Exception as e:
        logger.error(f"Error creating counselor thread: {str(e)}")
        return None

async def create_agent(coral_tools: List[Any]) -> AgentExecutor:
    coral_tools_description = get_tools_description(coral_tools)
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"""You are a suicide hotline triage assistant. Your job is to conduct a brief safety assessment and then connect callers to trained counselors using the available coral tools.

CORE RULES:
1. Speak with a calm, empathetic, supportive tone
2. Keep sentences short and clear — no jargon, no complex words
3. Follow this assessment process:
   - Greeting and empathy
   - Ask: "Are you safe right now?"
   - Ask: "Do you have thoughts of hurting yourself?"
   - Ask: "Do you have a plan to act on those thoughts?"
   - Summarize and prepare for handoff
4. Do not give medical, coping, or therapy advice
5. After assessment, use coral tools to connect with counselors

AVAILABLE CORAL TOOLS:
{coral_tools_description}

ASSESSMENT FLOW:
1. Greet caller warmly and ask safety questions
2. Based on responses, determine risk level
3. Use list_agents to find available counselors
4. Use create_thread to set up counselor connection
5. Use send_message to provide context to counselors
6. Complete handoff

RISK HANDLING:
- High Risk: "I'm connecting you immediately with a trained counselor."
- Low Risk: "I'm glad you're safe. Let me connect you with a counselor to talk further."
- Uncertain: "Let me connect you with a counselor who can provide better support."

RESPONSE STRUCTURE:
1. Greeting: "Hi, thank you for reaching out. I'm here to listen and ask a few quick questions before connecting you with a counselor."
2. Safety questions (ask one at a time)
3. Empathetic summary
4. Use coral tools to facilitate handoff
5. Confirm connection is being made

Note: discord agent is used to notify counselors in their Discord channel. so automatically notify when there is a risk or an emergency. send message in discord if you find out response from discord agent that informed to conselors.

Remember: Your role is brief triage and handoff, not extended counseling.
            """
        ),
        ("human", "{user_input}\n\nChat History:\n{chat_history}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    model = init_chat_model(
        model=os.getenv("MODEL_NAME"),
        model_provider=os.getenv("MODEL_PROVIDER"),
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=float(os.getenv("MODEL_TEMPERATURE", DEFAULT_TEMPERATURE)),
        max_tokens=int(os.getenv("MODEL_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
        base_url=os.getenv("MODEL_BASE_URL", None)
    )

    agent = create_tool_calling_agent(model, coral_tools, prompt)
    executor = AgentExecutor(agent=agent, tools=coral_tools, verbose=True, return_intermediate_steps=True)
    
    return executor

async def main():
    """Main function to run the triage agent with coral tools and terminal input."""
    try:
        config = load_config()

        coral_params = {
            "agentId": config["agent_id"],
            "agentDescription": "Suicide hotline triage assistant that provides initial safety assessment and connects callers to trained counselors"
        }
        
        query_string = urllib.parse.urlencode(coral_params)
        coral_server_url = f"{config['coral_sse_url']}?{query_string}"
        logger.info(f"Connecting to Coral Server: {coral_server_url}")

        timeout = float(os.getenv("TIMEOUT_MS", "30000"))
        
        client = MultiServerMCPClient(
            connections={
                "coral": {
                    "transport": "sse",
                    "url": coral_server_url,
                    "timeout": timeout,
                    "sse_read_timeout": timeout,
                }
            }
        )
        logger.info("Coral Server connection established")

        coral_tools = await client.get_tools(server_name="coral")
        logger.info(f"Retrieved {len(coral_tools)} coral tools")

        # Verify we have the coral tools we need
        required_coral_tools = [LIST_AGENTS_TOOL, CREATE_THREAD_TOOL, SEND_MESSAGE_TOOL, WAIT_FOR_MENTIONS_TOOL]
        available_tools = [tool.name for tool in coral_tools]
        
        for tool_name in required_coral_tools:
            if tool_name not in available_tools:
                error_message = f"Required coral tool '{tool_name}' not found"
                logger.error(error_message)
                raise ValueError(error_message)
        
        agent_tools = {tool.name: tool for tool in coral_tools}
        agent_executor = await create_agent(coral_tools)
        logger.info("Triage agent executor created")

        # Initialize user input handler
        user_input_handler = UserInputHandler()
        input_thread = user_input_handler.start_input_thread()

        chat_history: List[Dict[str, str]] = []
        current_thread_id = None

        # Send initial greeting
        initial_greeting = "Crisis Support Line - I'm here to listen. Can you tell me what's going on today?"
        print(f"Triage Agent: {initial_greeting}")

        try:
            while user_input_handler.running:
                try:
                    # Wait for user input (terminal has priority over coral messages)
                    user_input = await wait_for_user_message(agent_tools, user_input_handler, current_thread_id)
                    
                    if user_input is None:
                        # User requested to quit
                        break
                    
                    if user_input == "No message received":
                        # No input from either source, continue waiting
                        await asyncio.sleep(SLEEP_INTERVAL)
                        continue
                    
                    formatted_history = format_chat_history(chat_history)
                    
                    # Process the input through the agent
                    result = await agent_executor.ainvoke({
                        "user_input": user_input,
                        "agent_scratchpad": [],
                        "chat_history": formatted_history
                    })
                    
                    response = result.get('output', 'I apologize, but I need to connect you with a counselor right away. Please hold on.')
                    
                    # Send response (displays in terminal and/or sends to coral thread)
                    await send_triage_response(agent_tools, response, current_thread_id)

                    # Update chat history
                    chat_history.append({"user_input": user_input, "response": response})
                    
                    if len(chat_history) > MAX_CHAT_HISTORY:
                        chat_history.pop(0)
                    
                    # Check if we should create a counselor thread based on the conversation
                    if len(chat_history) >= 2:  # After a few exchanges, prepare for handoff
                        if not current_thread_id:
                            context_summary = f"Caller assessment: {chat_history[-1]['user_input'][:200]}..."
                            current_thread_id = await create_counselor_thread(agent_tools, context_summary)
                            
                            if current_thread_id:
                                handoff_message = "I'm now connecting you with a trained counselor who can provide you with the support you need. They'll be with you shortly."
                                await send_triage_response(agent_tools, handoff_message, current_thread_id)
                    
                    print()  # Add blank line for readability
                    await asyncio.sleep(SLEEP_INTERVAL)
                    
                except Exception as e:
                    logger.error(f"Error in triage agent loop: {str(e)}")
                    await asyncio.sleep(ERROR_RETRY_INTERVAL)
                    
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received, shutting down...")
        finally:
            user_input_handler.stop()
            print("Triage agent stopped. Take care.")
                
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())