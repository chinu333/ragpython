from mcp.server.fastmcp import FastMCP
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from geopy.geocoders import Nominatim
import requests
import yfinance as yf

# mcp = FastMCP(
#     name="Math",
#     host="127.0.0.1",
#     port=3030,
#     timeout=30
# )

env_path = os.path.dirname(os.path.dirname(__file__)) + os.path.sep + 'secrets.env'
load_dotenv(dotenv_path=env_path)

openaikey = os.getenv("AZURE_OPENAI_API_KEY")
openaiendpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openapideploymentname = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT_NAME")
aiapiversion = os.getenv("AZURE_OPENAI_API_VERSION")

azuremapssubskey = os.getenv("AZURE_MAPS_SUBSCRIPTION_KEY")
azuremapsclientid = os.getenv("AZURE_MAPS_CLIENT_ID")
print("Azure Maps Subscription Key: ", azuremapssubskey)
print("Azure Maps Client ID: ", azuremapsclientid)

llm = AzureChatOpenAI(
    azure_deployment=openapideploymentname,
    azure_endpoint=openaiendpoint,
    openai_api_key=openaikey,
    api_version=aiapiversion,
    verbose=False,
    temperature=0,
)

mcp = FastMCP("MCPServer")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    print("Adding", a, b)
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    print("Multiplying", a, b)
    return a * b

@mcp.tool()
def get_weather_info(location):
    """Get the weather info for location."""
    address=location
    geolocator = Nominatim(user_agent="Dummy")
    location = geolocator.geocode(address)
    lat = str(location.latitude)
    lon = str(location.longitude)

    # return "Weather is fantastic at " + location

    latlon = lat + "," + lon

    url = "https://atlas.microsoft.com/weather/currentConditions/json?api-version=1.0&query=" + latlon + "&subscription-key=" + azuremapssubskey

    headers = {
        'Content-Type': 'application/json',
        'x-ms-client-id': azuremapsclientid        
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        print("Status Code:", response.status_code)
        print("Response Body:", response.json())
        return response.content

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return f"An error occurred: {e}"

if __name__ == "__main__":
    print("Starting Math Server")
    mcp.run(transport="stdio")