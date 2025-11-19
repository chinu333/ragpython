from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from pathlib import Path  
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import asyncio
import nest_asyncio

# Apply nest_asyncio to allow nested event loops (needed for Jupyter/interactive environments)
nest_asyncio.apply()

class MCPClient:
    """
    Client for interacting with the MCP server.
    Provides methods to initialize the connection and execute user prompts.
    """
    
    def __init__(self):
        """Initialize the MCPClient with Azure OpenAI configuration."""
        # Load environment variables
        env_path = os.path.dirname(os.path.dirname(__file__)) + os.path.sep + 'secrets.env'
        load_dotenv(dotenv_path=env_path)
        
        # Get Azure OpenAI configuration
        self.openai_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.openai_deployment_name = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT_NAME")
        self.openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        
        # Initialize LLM
        self.llm = AzureChatOpenAI(
            azure_deployment=self.openai_deployment_name,
            azure_endpoint=self.openai_endpoint,
            openai_api_key=self.openai_key,
            api_version=self.openai_api_version,
            verbose=False,
            temperature=0,
        )
        
        # Configure server parameters
        script_path = (Path(__file__).resolve().parent / "mcpserver.py").resolve()
        self.server_params = StdioServerParameters(
            command="python",
            args=["./mcputils/mcpserver.py"],
            # command=str(sys.executable),    # use the running python binary
            # args=[str(script_path)],
            # # optionally set cwd to the script's directory:
            # cwd=str(script_path.parent),
        )
        
        # Create or get event loop
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
    
    async def _run_tool_async(self, prompt):
        """
        Internal async method to run the math tool with the provided prompt.
        
        Args:
            prompt: user prompt
            
        Returns:
            The response from the mcp server
        """
        # Using proper async with syntax to handle context managers correctly
        async with stdio_client(self.server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()
                
                # Get tools
                tools = await load_mcp_tools(session)
                
                # Create and run the agent
                agent = create_react_agent(self.llm, tools)
                agent_response = await agent.ainvoke({"messages": prompt})
                
                # Extract the final response
                msg_count = len(agent_response["messages"])
                msglist = agent_response["messages"]
                response_text = msglist[msg_count - 1]
                final_response = response_text.content
                
                return final_response
           
    def process_prompt(self, prompt):
        """
        Public method to execute a prompt.
        
        Args:
            prompt: user prompt
            
        Returns:
            The response from the server
        """
        # Handle the case when called from another file
        try:
            # Try to use a policy that works when calling from different contexts
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
            
            # Create a new event loop for each operation to avoid conflicts
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Run the async operation and return the result
                return loop.run_until_complete(self._run_tool_async(prompt))
            finally:
                # Clean up
                loop.close()
        except NotImplementedError:
            # Fall back to a simpler approach if the above fails
            print("Warning: Using fallback asyncio method")
            return asyncio.run(self._run_tool_async(prompt))

# Add a simple singleton instance for easier importing
_client_instance = None

def get_mcp_client():
    """Get a singleton instance of MCPClient."""
    global _client_instance
    if (_client_instance is None):
        _client_instance = MCPClient()
    return _client_instance

def execute_prompt(prompt):
    """
    Simple function to perform task based on the prompt.
    
    Args:
        prompt: User prompt
        
    Returns:
        The response from the server
    """
    return get_mcp_client().process_prompt(prompt)

# Example usage
if __name__ == "__main__":
    result = execute_prompt("what's (3 + 5) x 12?")
    print(f"Result: {result}")
    # result = execute_prompt("How is the weather in Kolkata today?")
    # print(f"Weather Result: {result}")
    # result = execute_prompt("Execute a quantum job with 80 repetitions.")
    # print(f"Quantum Process Result: {result}")

    result = execute_prompt("Could you please update me with the flight status of DL235?")
    print(f"Flight Status Result: {result}")