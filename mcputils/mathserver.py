from mcp.server.fastmcp import FastMCP

# mcp = FastMCP(
#     name="Math",
#     host="127.0.0.1",
#     port=3030,
#     timeout=30
# )

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    print("Adding", a, b)
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    print("Multiplying", a, b)
    return a * b

if __name__ == "__main__":
    print("Starting Math Server")
    mcp.run(transport="stdio")