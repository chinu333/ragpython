from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from pathlib import Path  
import os
from dotenv import load_dotenv


env_path = Path('.') / 'secrets.env'
load_dotenv(dotenv_path=env_path)

phiendpoint = os.getenv("AZURE_PHI_ENDPOINT")
phiapikey = os.getenv("AZURE_PHI_API_KEY")


def ask_phi(prompt):
    api_key = phiapikey
    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")

    client = ChatCompletionsClient(
        endpoint=phiendpoint,
        credential=AzureKeyCredential(api_key)
    )

    model_info = client.get_model_info()
    # print("Model name:", model_info.model_name)
    # print("Model type:", model_info.model_type)
    # print("Model provider name:", model_info.model_provider_name)

    payload = {
        "messages": [
            {
            "role": "user",
            "content": prompt
            }
        ],
        "max_tokens": 2048,
        "temperature": 0,
        "top_p": 0.1,
        "presence_penalty": 0,
        "frequency_penalty": 0
    }
    response = client.complete(payload)

    # print("Response:", response.choices[0].message.content)
    # print("Model:", response.model)
    # print("Usage:")
    # print("	Prompt tokens:", response.usage.prompt_tokens)
    # print("	Total tokens:", response.usage.total_tokens)

    return response.choices[0].message.content

# print("\n\n", ask_phi("What is the capital of France?"))