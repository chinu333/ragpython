from azure.search.documents.indexes.models import *
from pathlib import Path  
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAI
import base64
from urllib.request import urlopen
from langchain.schema.messages import HumanMessage, AIMessage



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

llm1 = AzureOpenAI(
    azure_deployment=openapideploymentname,
    azure_endpoint=openaiendpoint,
    openai_api_key=openaikey,
    api_version=aiapiversion,
    verbose=False,
    temperature=0,
)

def encode_image(image_url):
    return base64.b64encode(urlopen(image_url).read())

def analyze_image(question, image_url):
    print("Question :: " + question)
    print("Image URL :: " + image_url)

    messages = [
        (
            "system",
            "You are a helpful assistant that analyzes image and answers user questions. Restrict your response within 500 tokens. Please be polite and professional.",
        ),
        ("human", question, "![image](", image_url, ")"),
    ]

    questBody = "{\"temperature\":0,\"max_tokens\":1500,\"top_p\":1.0,\"messages\":[{\"role\":\"system\",\"content\":\"You are a helpful assistant that analyzes image.\"},{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"", question, "\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"", image_url, "\"}}]}]}"
    
    image = encode_image(image_url)

    message = HumanMessage(
        content=[
            {"type": "text", "text": "Analyze the image."},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    )


    # response = llm1.invoke(
    #     [   AIMessage(
    #         content="You are a helpful assistant that analyzes image and answers user questions."
    #     ),
    #         HumanMessage(
    #             content=[
    #                 {"type": "text", "text": {question}},
    #                 {
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": f"data:image/jpeg;base64,{image}"

    #                     },
    #                 },
    #             ]
    #         )
    #     ]
    # )
    
    response = llm1.invoke([message])
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