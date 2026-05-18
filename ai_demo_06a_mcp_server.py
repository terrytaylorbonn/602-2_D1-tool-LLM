# ai_demo_06a_mcp_server.py

# mcp_server_demo.py
# Minimal MCP server demo

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("DemoTools")

# -----------------------------------
# Tool
# -----------------------------------

@mcp.tool()
def read_shipment_status(truck_id: str) -> str:
    if truck_id == "truck_12":
        return "Truck 12 delayed in Taipei due to flooding."

    return "No status found."

# -----------------------------------

if __name__ == "__main__":
    mcp.run()