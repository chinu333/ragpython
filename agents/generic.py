from azure.search.documents.indexes.models import *
from pathlib import Path  
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache



env_path = Path('.') / 'secrets.env'
load_dotenv(dotenv_path=env_path)

openaikey = os.getenv("AZURE_OPENAI_API_KEY")
openaiendpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openapideploymentname = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT_NAME")
aiapiversion = os.getenv("AZURE_OPENAI_API_VERSION")

llm = AzureChatOpenAI(
    azure_deployment=openapideploymentname,
    azure_endpoint=openaiendpoint,
    openai_api_key=openaikey,
    api_version=aiapiversion,
    verbose=False,
    temperature=0,
)

cache = InMemoryCache()
set_llm_cache(cache)

def ask_generic_question(question):

    messages = [
        (
            "system",
            "You are a helpful assistant that answers user questions. Restrict your response within 500 tokens. Please be polite and professional.",
        ),
        ("human", question),
    ]
    response = llm.invoke(messages)
    # print("Question :: " + question)
    # print("AI Aisstant :: " + response.content + "\n")
    return response.content

# while True:
#     user_input = input("\nUser Question (or 'q' to quit): ")

#     if user_input == 'q':
#         break

#     try:
#         question = str(user_input)
#         ask_question(question)
#     except ValueError:
#         print("Invalid input. Please enter a number or 'q'.")