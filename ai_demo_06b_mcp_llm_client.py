# ai_demo_06b_mcp_llm_client

# ai_demo_06b_mcp_llm_client.py
# LLM chooses an MCP tool.
# MCP executes the tool.
# Code controls execution.

import asyncio
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def main():

    server_params = StdioServerParameters(
        command="python",
        args=["ai_demo_06b_mcp_llm_server.py"]
    )

    async with stdio_client(server_params) as streams:
        async with ClientSession(streams[0], streams[1]) as session:

            await session.initialize()

            tools = await session.list_tools()

            print("\nMCP TOOLS:")
            print(tools)

            user_prompt = "What is the condition of truck 12?"

            system_prompt = """
You are an AI agent.

You may call MCP tools.

Return ONLY JSON in this format:

{
  "tool": "read_shipment_status",
  "arguments": {
    "truck_id": "truck_12"
  }
}

Available MCP tool:
read_shipment_status(truck_id: str)
"""

            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )

            llm_output = response.choices[0].message.content

            print("\nLLM OUTPUT:")
            print(llm_output)

            action = json.loads(llm_output)

            tool_name = action["tool"]
            arguments = action["arguments"]

            result = await session.call_tool(tool_name, arguments)

            print("\nMCP TOOL RESULT:")
            print(result)


asyncio.run(main())