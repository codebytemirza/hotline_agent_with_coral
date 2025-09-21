import asyncio
import logging
from langchain_mcp_adapters import MCPClient

logging.basicConfig(level=logging.DEBUG)  # enable debug logs

async def main():
    try:
        client = MCPClient()

        # Ensure the server is registered properly
        # Example: "coral" server mapped to your backend URL
        await client.register_server(
            name="coral",
            url="http://localhost:8000",  # 🔹 replace with your real coral server URL
            headers={"Authorization": "Bearer YOUR_API_KEY"}  # if needed
        )

        logging.info("Attempting to fetch tools from 'coral' server...")
        coral_tools = await client.get_tools(server_name="coral")
        logging.info("Successfully connected to Coral server.")

        # Use the tools
        for tool in coral_tools:
            print(f"Loaded tool: {tool.name}")

    except Exception as e:
        logging.exception("❌ Failed to connect to Coral server.")
        raise

if __name__ == "__main__":
    asyncio.run(main())
