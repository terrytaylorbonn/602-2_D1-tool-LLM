# ai_demo_06c_mcp_llm_server.py

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("DemoTools")


@mcp.tool()
def read_shipment_status(truck_id: str) -> str:
    """Use this tool to get shipment status. Valid truck_id values: truck_12, truck_18."""

    if truck_id == "truck_12":
        return "Truck 12 is delayed in Taipei due to flooding."

    if truck_id == "truck_18":
        return "Truck 18 is on schedule in Taipei."

    return f"No shipment status found for {truck_id}."


@mcp.tool()
def read_supplier_status(supplier_id: str) -> str:
    """Use this tool to get supplier status. Valid supplier_id values: supplier_a, supplier_b."""

    if supplier_id == "supplier_a":
        return "Supplier A has an outage affecting brake components."

    if supplier_id == "supplier_b":
        return "Supplier B is operating normally."

    return f"No supplier status found for {supplier_id}."


if __name__ == "__main__":
    mcp.run()