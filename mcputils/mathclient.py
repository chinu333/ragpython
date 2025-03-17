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

class MathMCPClient:
    """
    Client for interacting with the MCP Math server.
    Provides methods to initialize the connection and run math operations.
    """
    
    def __init__(self):
        """Initialize the MathMCPClient with Azure OpenAI configuration."""
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
        self.server_params = StdioServerParameters(
            command="python",
            args=["./mcputils/mathserver.py"],
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
            prompt: The math problem to solve
            
        Returns:
            The response from the math server
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
                print(f"Final Answer :: =====>>>>> {final_response}")
                
                return final_response
    
    def run_math_operation(self, prompt):
        """
        Public method to solve a math problem.
        
        Args:
            prompt: The math problem to solve
            
        Returns:
            The solution to the math problem
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

def get_math_client():
    """Get a singleton instance of MathMCPClient."""
    global _client_instance
    if (_client_instance is None):
        _client_instance = MathMCPClient()
    return _client_instance

def run_math_problem(prompt):
    """
    Simple function to solve a math problem without needing to create a client.
    This makes it easier to import and use from other files.
    
    Args:
        prompt: The math problem to solve
        
    Returns:
        The solution to the math problem
    """
    return get_math_client().run_math_operation(prompt)

# Example usage
# if __name__ == "__main__":
#     result = run_math_problem("what's (3 + 5) x 12?")
#     print(f"Result: {result}")