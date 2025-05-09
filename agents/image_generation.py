# Note: DALL-E 3 requires version 1.0.0 of the openai-python library or later
import os
from openai import AzureOpenAI
import json
from pathlib import Path  
import os
from dotenv import load_dotenv
import requests
import base64
from PIL import Image
from io import BytesIO


env_path = os.path.dirname(os.path.dirname( __file__ )) + os.path.sep + 'secrets.env'
load_dotenv(dotenv_path=env_path)

openaikey = os.getenv("AZURE_OPENAI_API_KEY")
openaiendpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
aiapiversion = os.getenv("AZURE_OPENAI_API_VERSION")

dallekey = os.getenv("AZURE_OPENAI_DALL_E_ENDPOINT_API_KEY")
dalleendpoint = os.getenv("AZURE_OPENAI_DALL_E_ENDPOINT")
dalleapiversion = os.getenv("AZURE_OPENAI_DALL_E_API_VERSION")
dalle3deploymentname = os.getenv("AZURE_OPENAI_DALL_E_DEPLOYMENT_NAME")

# print("DALL-E 3 Deployment Name: ", dalle3deploymentname)
# print("DALL-E 3 Endpoint: ", dalleendpoint)
# print("DALL-E 3 API Version: ", dalleapiversion)
# print("DALL-E 3 Key: ", dallekey)
# print("OpenAI API Key: ", openaikey)

client = AzureOpenAI(
    api_version=aiapiversion,
    azure_endpoint=openaiendpoint,
    api_key=openaikey,
)

def generate_image_dalle3(prompt):

    result = client.images.generate(
        model=dalle3deploymentname, # the name of your DALL-E 3 deployment
        prompt=prompt,
        n=1
    )

    image_url = json.loads(result.model_dump_json())['data'][0]['url']
    print(result.model_dump_json())
    return image_url

def decode_and_save_image(b64_data, output_filename):
  image = Image.open(BytesIO(base64.b64decode(b64_data)))
#   image.show()
  image.save("data/" + output_filename)

def save_all_images_from_response(response_data, filename_prefix):
  for idx, item in enumerate(response_data['data']):
    b64_img = item['b64_json']
    filename = f"{filename_prefix}_{idx+1}.png"
    decode_and_save_image(b64_img, filename)
    print(f"Image saved to: '{filename}'")

def generate_image(prompt):
    base_path = f'openai/deployments/{dalle3deploymentname}/images'
    params = f'?api-version={dalleapiversion}'

    generation_url = f"{dalleendpoint}{base_path}/generations{params}"
    generation_body = {
        "prompt": prompt,
        "n": 1,
        "size": "1536x1024",
        "quality": "high",
        "output_format": "png"
    }
    generation_response = requests.post(
        generation_url,
        headers={
            'Api-Key': dallekey,
            'Content-Type': 'application/json',
        },
        json=generation_body
    ).json()

    # print("Generation Response: ", generation_response)

    save_all_images_from_response(generation_response, "generated_image")
    return "Image Generated Successfully."

# print(generate_image("Generate an image of an oil painting by Vincent Van Gogh depicting a garden with cherry blossoms by the river."))