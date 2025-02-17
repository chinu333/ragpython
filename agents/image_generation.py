# Note: DALL-E 3 requires version 1.0.0 of the openai-python library or later
import os
from openai import AzureOpenAI
import json
from pathlib import Path  
import os
from dotenv import load_dotenv


env_path = Path('.') / 'secrets.env'
load_dotenv(dotenv_path=env_path)

openaikey = os.getenv("AZURE_OPENAI_API_KEY")
openaiendpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
aiapiversion = os.getenv("AZURE_OPENAI_API_VERSION")
dalle3deploymentname = os.getenv("AZURE_OPENAI_DALL_E_DEPLOYMENT_NAME")

client = AzureOpenAI(
    api_version=aiapiversion,
    azure_endpoint=openaiendpoint,
    api_key=openaikey,
)

def generate_image(prompt):

    result = client.images.generate(
        model=dalle3deploymentname, # the name of your DALL-E 3 deployment
        prompt=prompt,
        n=1
    )

    image_url = json.loads(result.model_dump_json())['data'][0]['url']
    print(result.model_dump_json())
    return image_url

# print(generate_image("Generate a widescreen UX/UI website wireframe mockup for CSX, a Gofundme-like platform to help people subscribe to CSX Plus."))