# ai_demo_06c_mcp_llm_client.py
# MCP exposes tools dynamically.
# LLM chooses the correct tool.
# Code executes.

import asyncio
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def build_tools_text(tools):
    lines = []

    for tool in tools.tools:
        lines.append(f"- {tool.name}: {tool.description}")

    return "\n".join(lines)


async def main():

    server_params = StdioServerParameters(
        command="python",
        args=["ai_demo_06c_mcp_llm_server.py"]
    )

    async with stdio_client(server_params) as streams:
        async with ClientSession(streams[0], streams[1]) as session:

            await session.initialize()

            tools = await session.list_tools()
            tools_text = build_tools_text(tools)

            print("\nDYNAMIC MCP TOOL LIST:")
            print(tools_text)

            user_prompt = "What is the status of supplier A?"

            system_prompt = f"""
You are an AI agent.

Available MCP tools:

{tools_text}

Return ONLY JSON in this format:

{{
  "tool": "tool_name_here",
  "arguments": {{
    "argument_name_here": "argument_value_here"
  }}
}}
"""

            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0
            )

            llm_output = response.choices[0].message.content

            print("\nLLM OUTPUT:")
            print(llm_output)

            action = json.loads(llm_output)

            result = await session.call_tool(
                action["tool"],
                action["arguments"]
            )

            print("\nMCP TOOL RESULT:")
            print(result)


asyncio.run(main())