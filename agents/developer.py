from azure.search.documents.indexes.models import *
from pathlib import Path  
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from openai import AzureOpenAI



env_path = Path('.') / 'secrets.env'
load_dotenv(dotenv_path=env_path)

openaikey = os.getenv("AZURE_OPENAI_API_KEY")
openaiendpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openapideploymentname = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT_NAME")
aiapiversion = os.getenv("AZURE_OPENAI_API_VERSION")

# llm = AzureChatOpenAI(
#     azure_deployment=openapideploymentname,
#     azure_endpoint=openaiendpoint,
#     openai_api_key=openaikey,
#     api_version=aiapiversion,
#     verbose=False
# )

llm = AzureOpenAI(
  azure_endpoint = openaiendpoint, 
  api_key=openaikey,  
  api_version=aiapiversion
)

def generate_code(question):

    response = llm.chat.completions.create(
        model=openapideploymentname,
        messages=[
            {"role": "user", "content": question},
        ],
        max_completion_tokens = 10000

    )

    print(response.model_dump_json(indent=2))
    return response.choices[0].message.content 

# def generate_code(question):
#     messages = [
#         {"role": "user", "content": question},
#     ]
#     response = llm.invoke(messages)
#     # print("Question :: " + question)
#     # print("AI Aisstant :: " + response.content + "\n")
#     return response.content


# while True:
#     user_input = input("\nUser Question (or 'q' to quit): ")

#     if user_input == 'q':
#         break

#     try:
#         question = str(user_input)
#         generate_code(question)
#     except ValueError:
#         print("Invalid input. Please enter a number or 'q'.")