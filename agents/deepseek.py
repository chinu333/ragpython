
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.core.pipeline.transport import RequestsTransport
from pathlib import Path  
import os
from dotenv import load_dotenv


env_path = Path('.') / 'secrets.env'
load_dotenv(dotenv_path=env_path)

deepseekendpoint = os.getenv("AZURE_DEEPSEEK_ENDPOINT")
deepseekapikey = os.getenv("AZURE_DEEPSEEK_API_KEY")

def ask_deepseek(prompt):
    api_key = deepseekapikey

    if not api_key:
        raise Exception("A key should be provided to invoke the endpoint")

    client = ChatCompletionsClient(
        endpoint=deepseekendpoint,
        credential=AzureKeyCredential(api_key),
        transport=RequestsTransport(read_timeout=1200)
    )

    model_info = client.get_model_info()
    print("Model name:", model_info.model_name)
    print("Model type:", model_info.model_type)
    print("Model provider name:", model_info.model_provider_name)

    payload = {
    "messages": [
        {
        "role": "user",
        "content": prompt
        }
    ],
    "max_tokens": 8000
    }
    # response = client.complete(payload, stream=True)
    response = client.complete(payload)

    # print("Response:", response.choices[0].message.content)
    # print("Model:", response.model)
    # print("Usage:")
    # print("	Prompt tokens:", response.usage.prompt_tokens)
    # print("	Total tokens:", response.usage.total_tokens)
    # print("	Completion tokens:", response.usage.completion_tokens)

    return response.choices[0].message.content

# ask_deepseek("Write self contained code in html and javascript for a scientific calculator")
# ask_deepseek("What is the capital of France?")