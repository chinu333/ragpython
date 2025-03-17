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
from agents.rag import ask_vector_store
from agents.weather import get_weather_info
from agents.generic import ask_generic_question
from agents.financial_advisor import get_historical_stock_price
from agents.executesql import generate_response_from_sql
from agents.multimodality import analyze_image
from agents.visualization import get_dataframe
from agents.mermaid import generate_mermaid
from agents.notify import send_email
from agents.traffic import get_traffic_info
from agents.graphrag import ask_graph_db
from agents.developer import generate_code
from agents.search import search
from agents.image_generation import generate_image
from agents.nutrition import get_nutrition_info
from agents.phi import ask_phi
from agents.deepseek import ask_deepseek
from agents.space import get_space_info
from agents.speech import recognize_from_microphone
from dotenv import load_dotenv
from pathlib import Path  
import os
import matplotlib.pyplot as plt
from azure.core.credentials import AzureKeyCredential
from langchain_openai import AzureChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import seaborn as sns
from streamlit_mermaid import st_mermaid
import uuid
from mcputils.mathclient import run_math_problem

st.set_page_config(layout="wide",page_title="Agentic Copilot Demo")
st.set_option('deprecation.showPyplotGlobalUse', False)
styl = f"""
<style>
    .stTextInput {{
      position: fixed;
      bottom: 3rem;
    }}
</style>
"""
st.markdown(styl, unsafe_allow_html=True)
# Sidebar contents
with st.sidebar:

    st.title('AI Copilot With Agents')
    st.markdown('''
    ''')

    add_vertical_space(4)
    if st.button('Clear Chat'):
        st.markdown('')
        if 'history' in st.session_state:
            st.session_state['history'] = []
        if 'display_data' in st.session_state:
            st.session_state['display_data'] = {}


    st.markdown("""
                
### Sample Questions:
  
1. RAG: What was Microsoft\'s cloud revenue for 2024?
2. Compare Lucid and Tesla stocks and provide a recommendation for which one to buy.
3. Tell me something about Quantum Computing.
4. What are the total sales broken down by country? Show in a pie chart.
5. Please give me month-wise break up of the quantity of 'Chai' sold throughout the year of 2016.
6. Analyze the architecture diagram image and generate Terraform code for deploying all the resources in Azure. Please put all the resource in one resource group and the use the name rg_agents for the resource group. Image URL: https://ragstorageatl.blob.core.windows.net/miscdocs/WAF.png
7. What kind of cloth I need to wear today? I am in Atlanta, GA. Please also suggest a couple of stores in Atlanta where I can buy the clothes.
                


          """)
    st.write('')
    st.write('')
    st.write('')

    st.markdown('#### Created by Chinmoy C., 2025')

user_input= st.chat_input("You:")
image_generated = False
mermaid_generated = False
generated_mermaid_code = ''

@tool
def mcp_agent(prompt):
    """
    Call MCP Server tools to answer questions.
    
    Args:
        prompt: The user prompt
    
    Returns:
        Result retuned by the MCP Server
    """
    return run_math_problem(prompt)

# Check and remove conflicting environment variable
if "OPENAI_API_BASE" in os.environ:
    del os.environ["OPENAI_API_BASE"]

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
    # Return answer from the vector store
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
    Tool to provide stock recommendation based on hstorical stock prices specified in the ticker symbol.
    
    Args:
        Ticker symbol of the company.
    
    Returns:
        json: Lat 2 years stock information.
    """
    # Return stock info for last 2 years for the ticker symbol
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
    # Return answer of the generic question
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
    # Return results from SQL DB
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
    # Return image analysis text
    return analyze_image(question, image_url)

@tool
def mermaid_agent(prompt):
    """
    Tool to generate mermaid js code based on the user prompt.
    
    Args:
        User prompt
    
    Returns:
        str: Generated mermaid code.
    """
    # Return mermaid diagram
    mermaid_code = generate_mermaid(prompt)
    mermaid_code = mermaid_code.replace('`', '')
    mermaid_code = mermaid_code.replace('mermaid', '')
    mermaid_code = mermaid_code.replace('stateDiagram-v2;', 'stateDiagram-v2')
    mermaid_code = mermaid_code.replace('(Active)', '')
    mermaid_code = mermaid_code.replace('(Passive)', '')
    # mermaid_code = re.sub(r'([a-zA-Z0-9_.-])', '', mermaid_code)
    print("Mermaid Code :: ", mermaid_code)
    global mermaid_generated
    global generated_mermaid_code
    mermaid_generated = True
    generated_mermaid_code = mermaid_code
    return mermaid_code

@tool
def email_agent(receiver_email, subject, email_body):
    """
    Tool to send email to the respective receiver.
    
    Args:
        Receiver email address, email subject and the email message
    
    Returns:
        str: Confirmation of the email sent.
    """
    # Send email to the receiver
    return send_email(receiver_email, subject, email_body)

@tool
def traffic_agent(start_address, end_address):
    """
    Tool to get traffic updates between two locations.
    
    Args:
        start address and end address
    
    Returns:
        json: Traffic updates in json format.
    """
    # Get traffic updates between start address and end address
    return get_traffic_info(start_address, end_address)

@tool
def nutrition_agent(prompt):
    """
    Tool to get nutrition information for different food items.
    
    Args:
        user prompt for the nutrition info.
    
    Returns:
        json: Nutrition info in json format.
    """
    # Get nutrition info based on the user prompt
    return get_nutrition_info(prompt)

@tool
def graphrag_agent(question):
    """
    Tool to retieve answer from graph db.
    
    Args:
        user question.
    
    Returns:
        str: Answer from the graph database.
    """
    # Return answer from graph db
    return ask_graph_db(question)

@tool
def deepseek_agent(question):
    """
     Tool to retieve answer using DEEPSEEK model.
    
    Args:
        user question.
    
    Returns:
        str: Answer from the DEEPSEEK model.
    """
    # Return user answer using DEEPSEEK model
    return ask_deepseek(question)

@tool
def phi_agent(question):
    """
    Tool to retieve answer using PHI model.
    
    Args:
        user question.
    
    Returns:
        str: Answer from the PHI model.
    """
    # Return user answer using PHI model
    return ask_phi(question)

@tool
def image_generation_agent(prompt):
    """
    Tool to generate image from the user prompt.
    
    Args:
        user prompt.
    
    Returns:
        str: generated image url.
    """
    image_url = generate_image(prompt)
    # Return generated image url
    return image_url

@tool
def developer_agent(prompt):
    """
    Tool to write code in different languages.
    
    Args:
        User prompt.
    
    Returns:
        str: Generated code.
    """
    # Return the generated code
    return generate_code(prompt)

@tool
def search_agent(prompt):
    """
    Tool to search in the web.
    
    Args:
        User prompt.
    
    Returns:
        str: Search results.
    """
    # Return search results
    return search(prompt)

@tool
def space_info_agent(prompt):
    """
     Tool to provide answer realated to space, NASA, moon, mars etc. 
    
    Args:
        user question.
    
    Returns:
        str: Answer from the space_info_agent.
    """
    # Return user answer using space info agent
    return get_space_info(prompt)

def speech_agent():
    """
     Tool to recognize speech from microphone. 
    
    Args:
        no arugement.
    
    Returns:
        str: The text after the conversion of the speech to text.
    """
    global user_input
    user_input = recognize_from_microphone()
   
    return user_input

@tool
def data_visualization_agent(question, chart_type):
    """
    Tool to visulalize graph mentioned by graph_type argument. We'll execute the SQL using sql_agent and return the DataFrame.
    
    Args:
        User question and the graph type
    
    Returns:
        shows the graph.
    """
    # Return data frame after executing the SQL query
    answer, df = get_dataframe(question)
    print("Chart Type :: ", chart_type)
    global image_generated
    if 'pie'.lower() in chart_type.lower():
        #specifying the figure to plot 
        fig, x = plt.subplots(figsize=(5,5))
        # fig.suptitle(title, fontsize=10)
        # Create the pie chart
        x.pie(df["TotalSales"], labels=df["Country"], autopct='%1.1f%%', textprops={'fontsize': 8})
        x.axis('equal')  # Equal aspect ratio ensures a circular pie chart
        # sns.lineplot(x='x', y='y', data=df, ax=x)

        fig.savefig('data/chart.png')
        image_generated = True
    elif 'bar'.lower() in chart_type.lower():
        #specifying the figure to plot 
        fig, x = plt.subplots(figsize=(12,5))
        # Remove dollar signs and convert to numeric
        # df['TotalSales'] = df['TotalSales'].replace('[\$,]', '', regex=True).astype(float)
        # df['TotalSales'] = df['TotalSales'].replace('[,]', '', regex=True).astype(float)
        x.bar(df["Country"], df["TotalSales"])
        x.set_xlabel('Country')
        x.set_ylabel('TotalSales')
        # Add x, y gridlines
        x.grid(visible=True, color ='grey',
                linestyle ='-.', linewidth = 0.5,
                alpha = 0.2)

        # Show top values 
        fig.savefig('data/chart.png')
        image_generated = True
    elif 'scatter'.lower() in chart_type.lower():
        #specifying the figure to plot 
        fig, x = plt.subplots(figsize=(12,5))
        x.scatter(df["Country"], df["TotalSales"])
        x.set_xlabel('Country')
        x.set_ylabel('TotalSales')
        # Add x, y gridlines
        x.grid(visible=True, color ='grey',
                linestyle ='-.', linewidth = 0.5,
                alpha = 0.2)

        # Show top values 
        fig.savefig('data/chart.png')
        image_generated = True
    else:
        answer = answer, "\n\n", "At this point we don't support this type of chart. Please try with pie, bar or scatter chart. Stay tuned for more updates."

    print("Data Visulalization Agent Answer :: ", answer)
    return answer


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
            - question related to vector store with their own data (RAG) along with the source of the data.
            - question related to weather
            - question related to financial advice
            - question related to SQL queries for the structured data
            - question related to data visualization. Just return the answer in text format.
            - question related to general information such as computing, entertainment, genenral knowledge etc.
            - question related to generating mermaid js code from user prompt. You explain the diagram in simple language as well.
            - request to send email to the respective receiver with the email address, email subject and the user prompt.
            - question related to traffic between start address and end address. You get the traffic updates in json format. Analyze the json and provide the information in text format.
            - question related to graph database with their own data (GRAPH RAG)
            - question or prompt to generate image
            - question or prompt to generate code in different languages e.g. Python, Java, C++, HTML, Javascript etc. Explain the code as well.
            - question or prompt to search in the web.
            - question related to nutrition info of food. You get the nutrition info in json format. Analyze the json and provide the primary nutrition information in text format.
            - question or prompt to specifically ask 'PHI' model. Please invoke 'phi_agent' for this.
            - question or prompt to specifically ask 'DEEPSEEK' model. Please invoke 'deepseek_agent' for this.
            - question or prompt related to space, NASA, moon, mars etc.
            - question or prompt to recognize speech from the microphone. Please invoke 'speech_agent' for this.
            - question or prompt to for MCP server. Please invoke 'mcp_agent' for this.

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
    data_visualization_agent,
    mermaid_agent,
    email_agent,
    traffic_agent,
    graphrag_agent,
    image_generation_agent,
    developer_agent,
    search_agent,
    nutrition_agent,
    phi_agent,
    deepseek_agent,
    space_info_agent,
    speech_agent,
    mcp_agent
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

# print(graph.get_graph().draw_mermaid())
st_mermaid(graph.get_graph().draw_mermaid(), key="flow", height="300px")

# import shutil
import uuid

# Let's create an example conversation a user might have with the assistant
user_questions = []

if user_input:
    user_questions = [user_input]
    st.markdown("$${\color{#1df9ef}Human:}$$")
    st.markdown(user_input)


thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        "thread_id": thread_id,
    }
}

_printed = set()
for question in user_questions:
    events = graph.stream(
        {"messages": ("user", question)}, config, stream_mode="values"
        # {"messages": ("user", question)}, config, stream_mode="updates"
    )

    finalresponse = ''

    for event in events:
        # print("Agent Event Response :: ", event, _printed, "\n")
        # print("Agent Printed Response :: ", AIMessage(event.get("messages")).json(), "\n")
        lastMessage = event.get("messages")[len(event.get("messages")) - 1]
        print("Last Message Length :: ", len(lastMessage.content))
        print("Last Message :: ", lastMessage.content)
        finalresponse = ''
        
        if len(lastMessage.content) < 25000 :
            # print(lastMessage.content, "\n")
            # event.get("messages")[-1].pretty_print()
            # print("\n")
            finalresponse = event.get("messages")[-1].content
            
    st.markdown("$${\color{#19fa0a}AI:}$$")
    print("Final Response :: ", finalresponse)       
    st.markdown(finalresponse, unsafe_allow_html=True)
    if image_generated:
        st.image('data/chart.png')
        image_generated = False
    if mermaid_generated:
        st_mermaid(generated_mermaid_code, key="mermaid", height="600px")
        mermaid_generated = False
        generated_mermaid_code = ''