from datetime import datetime, timedelta
from mcp.server.fastmcp import FastMCP
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from geopy.geocoders import Nominatim
import requests
import yfinance as yf
from azure.quantum import Workspace
from azure.quantum.cirq import AzureQuantumService
import cirq
from a2aclient.a2aclient import perform_action

# mcp = FastMCP(
#     name="Math",
#     host="127.0.0.1",
#     port=3030,
#     timeout=30
# )

env_path = os.path.dirname(os.path.dirname(__file__)) + os.path.sep + 'secrets.env'
load_dotenv(dotenv_path=env_path)

azuremapssubskey = os.getenv("AZURE_MAPS_SUBSCRIPTION_KEY")
azuremapsclientid = os.getenv("AZURE_MAPS_CLIENT_ID")
print("Azure Maps Subscription Key: ", azuremapssubskey)
print("Azure Maps Client ID: ", azuremapsclientid)

aviationstackapikey = os.getenv("AVIATION_STACK_API_KEY")
a2aserverurl = os.getenv("A2A_SERVER_URL")


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
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

@mcp.tool()
def get_flight_status(flight_number):
    """
    Get flight information for a flight number using AviationStack API.
    
    Args:
        flight_number: Flight number (e.g., "DL123").

    Returns:
        json: Flight information.
    """
    print("Flight Info: ", flight_number)
    
    url = f"https://api.aviationstack.com/v1/flights?access_key={aviationstackapikey}&flight_iata={flight_number}"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error fetching flight status: {response.status_code} - {response.text}")
    
@mcp.tool()
async def convert_currency(prompt):
    """
    Convert currency from USD to EUR.
    
    Args:
        promtp: Prompt for the currency conversion.

    Returns:
        str: Information after the conversion.
    """
     
    return await perform_action(userPrompt=prompt, agent=a2aserverurl)

if __name__ == "__main__":
    print("Starting Math Server")
    mcp.run(transport="stdio")