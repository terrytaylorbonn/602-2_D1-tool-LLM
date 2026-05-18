# ai_demo_06a_mcp_client.py

# mcp_client_demo.py
# Minimal MCP client demo

import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():

    server_params = StdioServerParameters(
        command="python",
        args=["ai_demo_06a_mcp_server.py"]
    )

    async with stdio_client(server_params) as streams:

        async with ClientSession(
            streams[0],
            streams[1]
        ) as session:

            await session.initialize()

            # -----------------------------------
            # List available MCP tools
            # -----------------------------------

            tools = await session.list_tools()

            print("\nAVAILABLE TOOLS:")
            print(tools)

            # -----------------------------------
            # Call MCP tool
            # -----------------------------------

            result = await session.call_tool(
                "read_shipment_status",
                {"truck_id": "truck_12"}
            )

            print("\nTOOL RESULT:")
            print(result)

asyncio.run(main())