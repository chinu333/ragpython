from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from dotenv import load_dotenv
from pathlib import Path  
import os
import matplotlib.pyplot as plt
from azure.core.credentials import AzureKeyCredential
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
import nasapy
import os
import pandas as pd
import datetime
from datetime import datetime
import uuid

env_path = os.path.dirname(os.path.dirname(__file__)) + os.path.sep + 'secrets.env'
load_dotenv(dotenv_path=env_path)

nasaapikey  = os.getenv("NASA_API_KEY")
print("NASA API KEY:: " + nasaapikey)

openaikey = os.getenv("AZURE_OPENAI_API_KEY")
openaiendpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openapideploymentname = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT_NAME")
aiapiversion = os.getenv("AZURE_OPENAI_API_VERSION")

nasa = nasapy.Nasa(key=nasaapikey)

@tool
def picture_of_the_day_agent():
    """
    Tool to retieve picture of the day using NASA API.
    
    Args:
        user question.
    
    Returns:
        str: Description of the image and the image url.
    """
    print("Picture of the day called :: ")
    return nasa.picture_of_the_day(datetime.today().strftime('%Y-%m-%d'))

@tool
def mars_weather_agent():
    """
    Tool to retieve image data collected by the Mars rovers Curiosity, Discovery and Spirit.
    
    Args:
        user question.
    
    Returns:
        json: Mars weather for last seven days. The data is returned in JSON format. read the JSON data and return the data in text format.
    """
    return nasa.mars_weather()

@tool
def mars_rover_agent(earth_date):
    """
    Tool to retieve mars weather using NASA API.
    
    Args:
        Earth date in the format YYYY-MM-DD.
    
    Returns:
        json: Mars rover  in JSON. read the JSON data and return the data in text format.
    """
    return nasa.mars_rover(earth_date)

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
            '''You are a helpful customer support assistant for Multiverse Inc.
            You get the following type of questions from them:
            - question related to NASA picture of the day
            - question related to mars weather
            - question related to mars rover data for a specific date

            After you are able to discern all the information, call the relevant tool. Depending on the question, you might need to call multiple agents to answer the question appropriately.
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
part_2_tools = [
    picture_of_the_day_agent,
    mars_weather_agent,
    mars_rover_agent,
]

# Bind the tools to the space_assistant's workflow
part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(part_2_tools)

builder = StateGraph(State)
builder.add_node("space_assistant", Assistant(part_1_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(part_2_tools))

builder.add_edge(START, "space_assistant")  # Start with the assistant
builder.add_conditional_edges("space_assistant", tools_condition)  # Move to tools after input
builder.add_edge("tools", "space_assistant")  # Return to assistant after tool execution
builder.add_edge("space_assistant", END) # End with the assistant

def get_space_info(prompt):
    
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)

    thread_id = str(uuid.uuid4())
    print("Thread ID :: ", thread_id)

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    events = graph.stream(
        {"messages": ("user", prompt)}, config, stream_mode="values"
    )

    response = ''

    for event in events:
        lastMessage = event.get("messages")[len(event.get("messages")) - 1]
        response = ''
        
        if len(lastMessage.content) < 20000 :
            response = event.get("messages")[-1].content
        else:
            response = "Response is too long :: ", len(lastMessage.content)
    # print("Response :: ", response)
    return response  

# print(get_space_info('What is the NASA picture of the day?'))
# print(get_space_info('How is the Mars weather? Provide the details for last 2 days.'))
