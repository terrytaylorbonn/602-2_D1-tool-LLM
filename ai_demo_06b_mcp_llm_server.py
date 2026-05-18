# ai_demo_06b_mcp_llm_server.py
# MCP server exposing shipment tools

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ShipmentTools")


@mcp.tool()
def read_shipment_status(truck_id: str) -> str:
    """Return shipment status for a truck."""

    if truck_id == "truck_12":
        return "Truck 12 is delayed in Taipei due to flooding."

    if truck_id == "truck_18":
        return "Truck 18 is on schedule in Taipei."

    return f"No status found for {truck_id}."


if __name__ == "__main__":
    mcp.run()