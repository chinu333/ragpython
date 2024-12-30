from azure.search.documents.indexes.models import *
from pathlib import Path  
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI



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

def generate_mermaid(prompt):

    messages = [
        (
            "system",
            """
            Your job is to generate mermaid js code based on the below
            {prompt} information only, dont use any other information. Just generate the code and nothing extra.
            
            Code:
            """,
        ),
        ("human", prompt),
    ]
    response = llm.invoke(messages)
    # print("Question :: " + prompt)
    # print("AI Aisstant :: " + response.content + "\n")
    return response.content

# while True:
#     user_input = input("\nUser Question (or 'q' to quit): ")

#     if user_input == 'q':
#         break

#     try:
#         prompt = str(user_input)
#         generate_mermaid(prompt)
#     except ValueError:
#         print("Invalid input. Please enter a number or 'q'.")