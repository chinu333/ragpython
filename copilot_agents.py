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
from agents.weather import get_weather_from_azure_maps
from agents.generic import ask_generic_question
from agents.financial_advisor import get_financial_advice
from agents.executesql import generate_response_from_sql
from agents.multimodality import analyze_image
from agents.visualization import get_dataframe
from agents.mermaid import generate_mermaid
from agents.notify import send_email
from agents.traffic import get_traffic_info
from agents.graphrag import ask_graph_db
from agents.developer import generate_code
from agents.search import search
from agents.cosmos import save_to_cosmos
from mcputils.mcpclient import execute_prompt
from agents.image_generation import generate_image
from agents.nutrition import get_nutrition_info
from agents.diagram.diagramagent import draw
from agents.phi import ask_phi
from agents.deepseek import ask_deepseek
from agents.space import get_space_info
from agents.speech import recognize_from_microphone
from agents.docintel import extract_text_from_doc
# from agents.azurecua import process_cua
from dotenv import load_dotenv
from pathlib import Path  
import os
import base64
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
from fileuploader import upload_binary_to_azure, upload_file_locally
from audio_recorder_streamlit import audio_recorder
from openai import AzureOpenAI
import asyncio
import azure.cognitiveservices.speech as speechsdk
import time
import logging
from azure.monitor.opentelemetry import configure_azure_monitor
from opentelemetry import trace
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
# from agents.videorag import ask_video_rag
# from evaluation import evaluate_agents

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
    whisperendpoint = os.getenv("AZURE_WHISPER_ENDPOINT")
    whisperkey = os.getenv("AZURE_WHISPER_KEY")
    whisperdeploymentname = os.getenv("AZURE_WHISPER_DEPLOYMENT_NAME")
    azurespeechkey = os.getenv("SPEECH_KEY")
    azurespeechregion = os.getenv("SPEECH_REGION")
    appinsightconnectionstring = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")  

    # configure_azure_monitor(
    #     connection_string=appinsightconnectionstring,
    #     logger_name="copilot.agents",
    #     live_metrics=True,
    #     enable_auto_collect_requests=True,
    #     disabled_offline_storage=True,
    # )

    # agenttracer = trace.get_tracer("copilot.agents")
    # with agenttracer.start_as_current_span("Agentic Framework") as at:
    #     at.set_attribute("id", "copilot_agents")
    #     at.set_attribute("tools ", "testing")

# logger = logging.getLogger("copilot.agents")

# Get background image as base64
@st.cache_data
def get_background_image():
    with open("./images/background.jpg", 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache_data
def get_sidebar_background_image():
    with open("./images/sidebar_background.jpg", 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# initialize chat history in streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_voice_input = ''

st.set_page_config(layout="wide",page_title="Agentic Copilot Demo", page_icon="🌱")
# st.set_option('deprecation.showPyplotGlobalUse', False)
styl = f"""
<style>
    .stTextInput {{
      position: fixed;
      bottom: 3rem;
    }}
    .stApp {{
            background-image: url("data:image/png;base64,{get_background_image()}");
            background-size: cover;
            background-attachment: fixed; /* Optional: Makes background fixed during scroll */
    }}
    .stSidebar {{
            background-image: url("data:image/png;base64,{get_sidebar_background_image()}");
            background-size: cover;
            background-attachment: fixed; /* Optional: Makes background fixed during scroll */
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
        st.session_state.chat_history = []
        if 'history' in st.session_state:
            st.session_state['history'] = []
        if 'display_data' in st.session_state:
            st.session_state['display_data'] = {}
    
    st.write('')
    
    audio_conversion = st.toggle('Enable Voice Chat', False)
    if audio_conversion:
        cols = st.columns(2)
        with cols[0]:
            speech_input = audio_recorder(icon_size='1x', neutral_color="red")
            
            client = AzureOpenAI(
                api_key=whisperkey,  
                api_version="2024-02-01",
                azure_endpoint = whisperendpoint
            )
            
            deployment_id = whisperdeploymentname #This will correspond to the custom name you chose for your deployment when you deployed a model."

            if speech_input :
                transcript = client.audio.transcriptions.create(
                    model=deployment_id, 
                    file=("audio.wav", speech_input),
                )
                
                audio_prompt = transcript.text
                user_voice_input = audio_prompt
                print("Audio Prompt :: ", audio_prompt)

    st.write('')
    st.write('')
    
    
    st.markdown("""
                
### Sample Questions:
  
1. RAG: What was Microsoft\'s cloud revenue for 2024?
2. Compare Lucid and Tesla stocks and provide a recommendation for which one to buy.
3. Tell me something about Quantum Computing.
4. What are the total sales broken down by country? Show in a pie chart.
5. Analyze the WAF image and generate Terraform code to deploy all resources in Azure within a single resource group named rg_agents. Choose from the following image names: WAF, Mango, Tax Form, Uploaded Image, Seattle or Space Needle.
6. What kind of cloth I need to wear today? I am in Atlanta, GA. Please also suggest a couple of stores in Atlanta where I can buy the clothes.
7. Execute a quantum job with 80 repetitions.
                


          """)
    st.write('')
    st.write('')
    st.write('')

    st.markdown('#### Created by Chinmoy C., 2026')

user_input= st.chat_input("You:",
    accept_file=True,
    file_type=["jpg", "jpeg", "png"],
)
image_generated = False
mermaid_generated = False
generated_mermaid_code = ''
multimodality = False
dalle_image_generated = False
azure_diagram_generated = False

@tool
def mcp_agent(prompt):
    """
    Call MCP Server tools to answer questions. It can answer on following questions:
     1. get weather information
     2. get aviation information (e.g., flight status, timetable etc.)
     3. convert currency (e.g., USD to EUR)
     4. Execute a quantum process using QPU (Quantum Processing Unit)
     5. Evaluate model responses e.g Retrieval, Groundedness, Relevance, Response Completeness etc.
     6. Generate video from user prompt.
    
    Args:
        prompt: The user prompt
    
    Returns:
        Result retuned by the MCP Server. Strip all the escape characters from the response.
    """        
    return execute_prompt(prompt)
    
# @tool
# def quantum_agent(repetitions_count):
#     """
#     Execute a quantum process using QPU (Quantum Processing Unit).
    
#     Args:
#         repetitions_count: Number of repetitions for the QPU process.

#     Returns:
#         json: QPU process result.
#     """
#     return submit_quantum_job(repetitions_count)

@tool
def cosmos_agent(prompt, response):
    """
    Tool to save user prompt and the final response into cosmos db.
    
    Args:
        prompt: The user prompt
        response: The final response from the agent
    
    Returns:
        Data saved to Cosmos DB confirmation.
    """
    return save_to_cosmos(prompt, response)

@tool
def architecture_diagram_agent(prompt):
    """
    Tool to draw architecture diagram in Azure.
    
    Args:
        prompt: The user prompt
    
    Returns:
        str: The path to the generated architecture diagram image.
    """

    drawn_diagram_path = draw(prompt)
    global azure_diagram_generated
    azure_diagram_generated = True
    return drawn_diagram_path

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
def docintel_agent(file_name):
    """
    Tool to extract text from a given file name.
    
    Args:
        file_name: The name of the file for which the data needs to be extracted
    
    Returns:
        The extracted text from the document.
    """
    return extract_text_from_doc(file_name)

# @tool
# def video_rag_agent(question):
#     """
#     Tool to retieve answer from video data stored in vector db.
    
#     Args:
#         user question.
    
#     Returns:
#         str: Answer from the vector store.
#     """
#     # Return answer from the vector store
#     return ask_video_rag(question)

# @tool
# def cua_agent(question):
#     """
#     Tool to execute Computer Use Agent (CUA) on the behalf of the user.
    
#     Args:
#         user prompt.
    
#     Returns:
#         str: Answer from the CUA agent.
#     """
#     # Return answer from the vector store
#     return process_cua(question)

# @tool
# def weather_agent(location):
#     """
#     Tool to retieve weather information.
    
#     Args:
#         Location for which the weather info is required.
    
#     Returns:
#         str: Weather Information.
#     """
#     # Return weather info based on the place
#     return get_weather_from_azure_maps(location)

@tool
def financial_advisor_agent(prompt):
    """
    Tool to compare and provide stock recommendation specified in the ticker symbol and/or prediction of the stock price for the future date.
    
    Args:
        Prompt for the financial advisor agent.
    
    Returns:
        str: Financial recommendation based on the prompt.
    """
    # Return stock info for last 2 years for the ticker symbol
    return get_financial_advice(prompt)

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
def analyze_image_agent(question, image_file_name):
    """
    Tool to analyze image mentioned in the image_file name.
    
    Args:
        User question and the image file name
    
    Returns:
        str: Answer of the user question.
    """
    # Return image analysis text
    global multimodality
    multimodality = True
    return analyze_image(question, image_file_name)

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
        str: Image generation result if the generation is successful.
    """
    generate_image(prompt)
    # Return generated image url
    global dalle_image_generated
    dalle_image_generated = True
    # return image_url

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

@tool
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
            '''You are a helpful customer support assistant for Contoso Inc. You have more than 15 agents to assist with various tasks. Invoke agent or agents depending on the question.
            You get the following type of questions from them:
            - question related to vector store with their own data (RAG) along with the source of the data. 
            - question related to vector store with video data (VIDEO RAG).
            - question related to weather. Respond with the weather information both in celcius and fahrenheit.
            - question related to comparing multiple stocks and provide recommendation. In this case, call 'financial_advisor_agent' and provide the recommendation and the analysis. Stock recommendation can be for any company(s). 
            - question related to prediction of the Microsoft stock price for the future date. Do not answer if the prediction is for any other company. Respond with the range of stock prices considering the MSE (Mean Squared Error) of the model. Format the 'disclaimer' word in RED color for 'streamlit' ui text.
            - question related to SQL queries for the structured data
            - question related to data visualization. Just return the answer in text format.
            - question related to general information such as computing, entertainment, genenral knowledge etc.
            - question related to generating mermaid js code from user prompt. You explain the diagram in simple language as well.
            - question related to generating architecture diagrams in Azure based on user prompt. User cam mention the terraform code as well.
            - request to send email to the respective receiver with the email address, email subject and the user prompt.
            - question related to traffic between start address and end address. You get the traffic updates in json format. Analyze the json and provide the information in text format.
            - question related to graph database with their own data (GRAPH RAG)
            - question related to analyzing images. Please use this info for image name to image_file name mapping ['WAF': 'WAF.png', 'Mango':'Mango_4.png', 'Tax Form':'Tax_Form.png', 'Seattle or Space Needle':'Space_Needle.png', 'Uploaded Image':'uploaded_image.png']
            - question or prompt to generate image
            - question or prompt to extract text from document and respond based on the user question (docintel_agent)
            - question or prompt to generate code in different languages e.g. Python, Java, C++, HTML, Javascript etc. Explain the code as well.
            - question or prompt to search in the web.
            - question related to nutrition info of food. The food item may be attached as an image. Invoke 'analyze_image' agent and pass on the text to 'nutrition_agent' to get the nutrition info. You get the nutrition info in json format. Analyze the json and provide the primary nutrition information in text format.
            - question or prompt to specifically ask 'PHI' model. Please invoke 'phi_agent' for this.
            - question or prompt to specifically ask 'DEEPSEEK' model. Please invoke 'deepseek_agent' for this.
            - question or prompt related to space, NASA, moon, mars etc.
            - question or prompt to recognize speech from the microphone. Please invoke 'speech_agent' for this.
            - question or prompt to for MCP server. Please invoke 'mcp_agent' for this. Remove all excape characters from the response.
            - question or prompt to for CUA (Computer Use Agent). Please invoke 'cua_agent' for this. Remove all escape characters from the response.
            - question or prompt related to execute a process on quantum processor.

            After you are able to discern all the information, call the relevant tool. Depending on the question, you might need to call multiple agents to answer the question appropriately. Call the generic agent by default. Invoke 'search-agent' if you don't get any relevant information from the 'generic_agent'.
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
    # weather_agent,
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
    mcp_agent,
    # cua_agent,
    cosmos_agent,
    # video_rag_agent,
    docintel_agent,
    architecture_diagram_agent,
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

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()

memory = st.session_state.memory

# if "graph" not in st.session_state:
#     st.session_state.graph = builder.compile(checkpointer=memory)

# graph = st.session_state.graph

# memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
# evaluate_agents(graph)

# print(graph.get_graph().draw_mermaid())
# mermaid_graph = graph.get_graph().draw_mermaid()
# mermaid_graph = mermaid_graph.replace('fill-opacity:0', 'fill-opacity:1')
# mermaid_graph = mermaid_graph + """
#     linkStyle default stroke-width:2px, stroke:#ffffff
#     linkStyle 0 stroke:#13fa04
#     linkStyle 3 stroke:#04faef
#     linkStyle 4 stroke:#ef04fa
#     """
# st_mermaid(mermaid_graph, key="flow", height="300px")

# import shutil
import uuid

def stream_data(finalresponse):
    for word in finalresponse.split(" "):
        yield word + " "
        time.sleep(0.02)

# display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Let's create an example conversation a user might have with the assistant
user_questions = []

if (user_input and user_input.text):
    user_questions = [user_input.text]
    st.markdown("$${\color{#1df9ef}Human:}$$")
    st.markdown(user_input.text)

    # st.chat_message("user").markdown(user_input.text)
    st.session_state.chat_history.append({"role":"user","content": user_input.text})

    # logger.info("User Input :: %s", user_input.text)

if user_voice_input:
    user_questions = [user_voice_input]
    st.markdown("$${\color{#1df9ef}Human:}$$")
    st.markdown(user_voice_input)
    # logger.info("User Voice Input :: %s", user_voice_input)

if user_input and user_input["files"]:
    file = user_input["files"][0]
    filename = file.name
    filetype = file.type
    print("File Name", file.name)
    print("File Type", file.type)
    if file is not None:
        # To read file as bytes:
        bytes_data = file.getvalue()
        # st.write(bytes_data)
        # binary_url = upload_binary_to_azure(
        #     bytes_data, 
        #     blob_name=filename,
        #     content_type=filetype, 
        #     container_name="miscdocs"
        # )
        # print("Binary URL :: ", binary_url)
        image_name = upload_file_locally(bytes_data)
        print("Image Name :: ", image_name)
        user_questions = [user_input.text + ". Image : " + image_name]

def get_thread_id():
    """
    Function to get the thread ID.
    
    Returns:
        str: The thread ID.
    """
    # Generate thread id if the chat history is empty
    if "thread_id" not in st.session_state:
        thread_id = str(uuid.uuid4())
        st.session_state.thread_id = thread_id
        return thread_id
    else:
        thread_id = st.session_state.thread_id
        return thread_id


config = {
    "configurable": {
        "thread_id": get_thread_id()
    }
}

cua = False
# async def process_langraph_cua(msg):
#     # Stream the graph execution
#     stream = cua_agent.astream(
#         {"messages": msg},
#         stream_mode="updates"
#     )

#     # Process the stream updates
#     async for update in stream:
#         if "create_vm_instance" in update:
#             print("VM instance created")
#             stream_url = update.get("create_vm_instance", {}).get("stream_url")
#             # Open this URL in your browser to view the CUA stream
#             print(f"Stream URL: {stream_url}")

#     print("Done")

_printed = set()
for question in user_questions:
    if cua:
        # asyncio.run(process_langraph_cua(question))
        print("Processing CUA...")

    else:
        events = graph.stream(
            {"messages": ("user", question)}, config, stream_mode="values"
            # {"messages": ("user", question)}, config, stream_mode="updates"
        )

        finalresponse = ''

        for event in events:
            # print("Agent Event Response :: ", event, _printed, "\n")
            # print("Agent Printed Response :: ", AIMessage(event.get("messages")).json(), "\n")
            lastMessage = event.get("messages")[len(event.get("messages")) - 1]
            # print("Last Message Length :: ", len(lastMessage.content))
            # print("Last Message :: ", lastMessage.content)
            # print("Number of msgs :: ", len(event.get("messages")) )

            bigResponse = ''

            for msg in event.get("messages"):
                if msg.content:
                    print("Agent Response Length :: +++++++ ", len(msg.content))
                    # print("Agent Response :: ==>>>>> ", msg.content)
                    if len(msg.content) > 5000 and multimodality:
                        bigResponse = msg.content

            finalresponse = ''
            
            if len(bigResponse) < 5000 :
                # print(lastMessage.content, "\n")
                # event.get("messages")[-1].pretty_print()
                # print("\n")
                finalresponse = event.get("messages")[-1].content
            else:
                finalresponse = bigResponse
                multimodality = False
                
        st.markdown("$${\color{#19fa0a}AI:}$$")
        print("Final Response :: ", finalresponse)

        # st.chat_message("user").markdown(finalresponse)
        st.session_state.chat_history.append({"role":"assistant","content": finalresponse})


        # logger.info("Final Response :: %s", finalresponse)
        # st.markdown(finalresponse, unsafe_allow_html=True)
        st.write_stream(stream_data(finalresponse))

        if user_voice_input:
            speech_config = speechsdk.SpeechConfig(subscription=azurespeechkey, region=azurespeechregion)
            file_name = "./audio/outputaudio.wav"
            file_config = speechsdk.audio.AudioOutputConfig(filename=file_name)
            audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
            speech_config.speech_synthesis_voice_name = "en-US-AvaMultilingualNeural"
            # speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
            speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=file_config)
            speech_sysnthesis_result = speech_synthesizer.speak_text_async(finalresponse)

            if speech_sysnthesis_result.get().reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                print("Speech synthesis was successful")
            else:
                print("Speech synthesis failed: {}".format(speech_sysnthesis_result.get().cancellation_details.error_details))

            # play_audio
            st.audio("./audio/outputaudio.wav", format="audio/wav",  loop=False, autoplay=True)
        
        if image_generated:
            st.image('data/chart.png')
            image_generated = False
        if  dalle_image_generated:
            st.image('data/generated_image_1.png')
            dalle_image_generated = False
        if mermaid_generated:
            st_mermaid(generated_mermaid_code, key="mermaid", height="600px")
            mermaid_generated = False
            generated_mermaid_code = ''
        if azure_diagram_generated:
            st.image('images/architecture_diagram.png')
            azure_diagram_generated = False
        # if video_generated:
        #     # st.video("data/video.mp4", format="video/mp4", start_time=0)
        #     video_generated = False
        #     print("Video generated successfully and saved as 'video.mp4'.")
        if 'generate_video' in st.session_state:
            st.video("data/video.mp4", format="video/mp4", start_time=0)
            st.session_state['generate_video'] = 'false'
            print("Video generated successfully and saved as 'video.mp4'.")
