from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_openai import AzureChatOpenAI
from pathlib import Path  
import os
from dotenv import load_dotenv

env_path = Path('.') / 'secrets.env'
load_dotenv(dotenv_path=env_path)

openaikey = os.getenv("AZURE_OPENAI_API_KEY")
openaiendpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openapideploymentname = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT_NAME")
aiapiversion = os.getenv("AZURE_OPENAI_API_VERSION")
os.environ["GOOGLE_API_KEY"] = "xxx"
os.environ["GOOGLE_CSE_ID"] = "yyy"


llm = AzureChatOpenAI(
    azure_deployment=openapideploymentname,
    azure_endpoint=openaiendpoint,
    openai_api_key=openaikey,
    api_version=aiapiversion,
    verbose=False,
    temperature=0,
)

def search(prompt):

    tools = load_tools(["google-search"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False
    )

    response = agent.run(prompt)
    # print("Question :: " + prompt)
    # print("AI Aisstant :: " + response + "\n")
    return response

# print("\n\n--------\n" + search("What are the top restaurants near Microsoft Atlanta office? Please provide their website link as well.") + "\n--------\n\n")