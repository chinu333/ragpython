from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from copilot import ask_vector_store
from weather import get_weather_info
from generic import ask_generic_question
from financial_advisor import get_historical_stock_price
from executesql import generate_response_from_sql
from multimodality import analyze_image
from dotenv import load_dotenv
from pathlib import Path  
import os
from azure.core.credentials import AzureKeyCredential
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from pydantic import BaseModel
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.human import HumanMessage
import python_weather


if __name__ == "__main__":

    env_path = Path('.') / 'secrets.env'
    load_dotenv(dotenv_path=env_path)

    aisearchindexname = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")
    aisearchkey = os.getenv("AZURE_AI_SEARCH_KEY")
    openaikey = os.getenv("AZURE_OPENAI_API_KEY")
    openaiendpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    aisearchendpoint= os.getenv("AZURE_AI_SEARCH_SERVICE_ENDPOINT")
    search_creds = AzureKeyCredential(aisearchkey)
    embeddingname = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    openapideploymentname = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT_NAME")
    aiapiversion = os.getenv("AZURE_OPENAI_API_VERSION")

@tool
def rag_agent(question):
    """
    Tool to retieve answer from custom data stored in vector db.
    
    Args:
        user question.
    
    Returns:
        str: Answer from the vector store.
    """
    # Return calculated solar savings
    return ask_vector_store(question)

@tool
def weather_agent(location):
    """
    Tool to retieve weather information.
    
    Args:
        Location for which the weather info is required.
    
    Returns:
        str: Weather Information.
    """
    # Return weather info based on the place
    return get_weather_info(location)

@tool
def financial_advisor_agent(ticker):
    """
    Tool to provide stock recommendation based on hstorical stock prices psecified in the ticker symbol.
    
    Args:
        Ticker symbol of the company.
    
    Returns:
        json: Lat 5 years stock information.
    """
    # Return weather info based on the place
    return get_historical_stock_price(ticker)

@tool
def generic_agent(question):
    """
    Tool to provide answer to the user generic question.
    
    Args:
        User question.
    
    Returns:
        str: Answer of the user question.
    """
    # Return weather info based on the place
    return ask_generic_question(question)

@tool
def sql_agent(question):
    """
    Tool to provide answer by running SQL query against structured data.
    
    Args:
        User question.
    
    Returns:
        str: Answer of the user question.
    """
    # Return weather info based on the place
    return generate_response_from_sql(question)

@tool
def analyze_image_agent(question, image_url):
    """
    Tool to analyze image mentioned in the image_url.
    
    Args:
        User question and the image url
    
    Returns:
        str: Answer of the user question.
    """
    # Return weather info based on the place
    return analyze_image(question, image_url)


def handle_tool_error(state) -> dict:
    """
    Function to handle errors that occur during tool execution.
    
    Args:
        state (dict): The current state of the AI agent, which includes messages and tool call details.
    
    Returns:
        dict: A dictionary containing error messages for each tool that encountered an issue.
    """
    # Retrieve the error from the current state
    error = state.get("error")
    
    # Access the tool calls from the last message in the state's message history
    tool_calls = state["messages"][-1].tool_calls
    
    # Return a list of ToolMessages with error details, linked to each tool call ID
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",  # Format the error message for the user
                tool_call_id=tc["id"],  # Associate the error message with the corresponding tool call ID
            )
            for tc in tool_calls  # Iterate over each tool call to produce individual error messages
        ]
    }

def create_tool_node_with_fallback(tools: list) -> dict:
    """
    Function to create a tool node with fallback error handling.
    
    Args:
        tools (list): A list of tools to be included in the node.
    
    Returns:
        dict: A tool node that uses fallback behavior in case of errors.
    """
    # Create a ToolNode with the provided tools and attach a fallback mechanism
    # If an error occurs, it will invoke the handle_tool_error function to manage the error
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)],  # Use a lambda function to wrap the error handler
        exception_key="error"  # Specify that this fallback is for handling errors
    )

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class Assistant:
    def __init__(self, runnable: Runnable):
        # Initialize with the runnable that defines the process for interacting with the tools
        self.runnable = runnable

    def __call__(self, state: State):
        while True:
            # Invoke the runnable with the current state (messages and context)
            result = self.runnable.invoke(state)
            
            # If the tool fails to return valid output, re-prompt the user to clarify or retry
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                # Add a message to request a valid response
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                # Break the loop when valid output is obtained
                break

        # Return the final state after processing the runnable
        return {"messages": result}
    
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            '''You are a helpful customer support assistant for Contoso Inc.
            You get the following type of questions from them:
            - question related to vector store with their own data (RAG)
            - question related to weather
            - question related to financial advice
            - question related to SQL queries for the structured data
            - question related to general information such as computing, entertainment, genenral knowledge etc.

            After you are able to discern all the information, call the relevant tool. Depending on the question, you might need to call multiple agents to answer the question appropriately. Call the generic agent by default.
            ''',
        ),
        ("placeholder", "{messages}"),
    ]
)

llm = AzureChatOpenAI(
    azure_deployment=openapideploymentname,
    azure_endpoint=openaiendpoint,
    openai_api_key=openaikey,
    api_version=aiapiversion,
    verbose=False,
    temperature=0,
)

# Define the tools the assistant will use
part_1_tools = [
    rag_agent,
    weather_agent,
    financial_advisor_agent,
    generic_agent,
    sql_agent,
    analyze_image_agent,
]

# Bind the tools to the assistant's workflow
part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(part_1_tools)

builder = StateGraph(State)
builder.add_node("assistant", Assistant(part_1_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))

builder.add_edge(START, "assistant")  # Start with the assistant
builder.add_conditional_edges("assistant", tools_condition)  # Move to tools after input
builder.add_edge("tools", "assistant")  # Return to assistant after tool execution
builder.add_edge("assistant", END) # End with the assistant

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# import shutil
import uuid

# Let's create an example conversation a user might have with the assistant
tutorial_questions = [
    # 'RAG: What was Microsoft\'s cloud revenue for 2024?',
    # 'What kind of cloth I need to wear today? I am in Atlanta, GA.',
    # 'Should I buy Microsoft stock?',
    # 'Compare Google and Tesla stocks and provide a recommendation for which one to buy.',
    # 'How I go from Atlanta to Disney World, FL?',
    # 'Tell me something about Quantum Computing.',
    # 'What are the total sales broken down by country?',
    # 'What are the top 10 most popular products based on quantity sold?',
    # 'Based on the historical stock price of Delta Airlines, please advise if I should buy the stock during holiday season when everyone travels?',
    # 'Analyze the architecture diagram image and generate Terraform code for deploying all the resources in Azure. Please put all the resource in one resource group and the use the name rgXXXX for the resource group. Image URL: https://ragstorageatl.blob.core.windows.net/miscdocs/WAF.png',
    'Analyze the image. Image URL: https://ragstorageatl.blob.core.windows.net/miscdocs/Space_Needle.png'
]


thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        "thread_id": thread_id,
    }
}

_printed = set()
for question in tutorial_questions:
    events = graph.stream(
        {"messages": ("user", question)}, config, stream_mode="values"
        # {"messages": ("user", question)}, config, stream_mode="updates"
    )

    for event in events:
        # kind = event["event"]
        # print("Agent Event Response :: ", event, _printed, "\n")
        # print("Agent Printed Response :: ", AIMessage(event.get("messages")).json(), "\n")
        lastMessage = event.get("messages")[len(event.get("messages")) - 1]
        
        if len(lastMessage.content) < 5000 :
            # print(lastMessage.content, "\n")
            event.get("messages")[-1].pretty_print()
            print("\n")