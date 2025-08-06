from azure.search.documents.indexes.models import *
from pathlib import Path  
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.schema.messages import HumanMessage
import base64
import httpx
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache
from mimetypes import guess_type


env_path = os.path.dirname(os.path.dirname( __file__ )) + os.path.sep + 'secrets.env'
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

cache = InMemoryCache()
set_llm_cache(cache)

def encode_image(image_url):
    return base64.b64encode(httpx.get(image_url).content).decode("utf-8")

def analyze_image(question, image_file_name):
    print("Question :: " + question)
    print("Image File :: " + image_file_name)
    image_path = './images/' + image_file_name
    data_url = local_image_to_data_url(image_path)
    # print("Data URL :: " + data_url)

    message = HumanMessage(
        content=[
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"{data_url}"}}
        ]
    )

    # print("Multimodality cache :: ", cache._cache)
    
    response = llm.invoke([message])
    return response.content


# Function to encode a local image into data URL 
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

# Example usage
# image_path = '../images/WAF.png'
# data_url = local_image_to_data_url(image_path)
# print("Data URL:", data_url)

# while True:
#     user_input = input("\nUser Question (or 'q' to quit): ")

#     if user_input == 'q':
#         break

#     try:
#         question = str(user_input)
#         analyze_image(question, "https://ragstorageatl.blob.core.windows.net/miscdocs/Space_Needle.png")
#     except ValueError:
#         print("Invalid input. Please enter a number or 'q'.")

# print(analyze_image("Analyze the image.", "../images/WAF.png"))