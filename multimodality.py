from azure.search.documents.indexes.models import *
from pathlib import Path  
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.schema.messages import HumanMessage
import base64
import httpx


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
    verbose=True,
    temperature=0,
)

def encode_image(image_url):
    return base64.b64encode(httpx.get(image_url).content).decode("utf-8")

def analyze_image(question, image_url):
    print("Question :: " + question)
    print("Image URL :: " + image_url)

    message = HumanMessage(
        content=[
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    )
    
    response = llm.invoke([message])
    return response.content

# while True:
#     user_input = input("\nUser Question (or 'q' to quit): ")

#     if user_input == 'q':
#         break

#     try:
#         question = str(user_input)
#         analyze_image(question, "https://ragstorageatl.blob.core.windows.net/miscdocs/Space_Needle.png")
#     except ValueError:
#         print("Invalid input. Please enter a number or 'q'.")