from mcp.server.fastmcp import FastMCP
import os
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
import requests
from azure.quantum import Workspace
from azure.quantum.cirq import AzureQuantumService
import cirq
import json
from a2aclient.a2aclient import perform_action
from azure.ai.evaluation import GroundednessEvaluator, AzureOpenAIModelConfiguration, evaluate, QAEvaluator, RelevanceEvaluator, RetrievalEvaluator

# mcp = FastMCP(
#     name="Math",
#     host="127.0.0.1",
#     port=3030,
#     timeout=30
# )

env_path = os.path.dirname(os.path.dirname(__file__)) + os.path.sep + 'secrets.env'
load_dotenv(dotenv_path=env_path)

env_path = os.path.dirname(os.path.dirname(__file__)) + os.path.sep
print("Environment Path: ", env_path)

azuremapssubskey = os.getenv("AZURE_MAPS_SUBSCRIPTION_KEY")
azuremapsclientid = os.getenv("AZURE_MAPS_CLIENT_ID")
# print("Azure Maps Subscription Key: ", azuremapssubskey)
# print("Azure Maps Client ID: ", azuremapsclientid)

aviationstackapikey = os.getenv("AVIATION_STACK_API_KEY")
a2aserverurl = os.getenv("A2A_SERVER_URL")
azure_quantum_conn_str = os.getenv("AZURE_QUANTUM_CONNECTION_STRING")
# print("Azure Quantum Connection String: ", azure_quantum_conn_str)
openaiendpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openaikey = os.getenv("AZURE_OPENAI_API_KEY")
openapideploymentname = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT_NAME")
aiapiversion = os.getenv("AZURE_OPENAI_API_VERSION")


mcp = FastMCP("MCPServer")

# @mcp.tool()
# def add(a: int, b: int) -> int:
#     """Add two numbers"""
#     print("Adding", a, b)
#     return a + b

# @mcp.tool()
# def multiply(a: int, b: int) -> int:
#     """Multiply two numbers"""
#     print("Multiplying", a, b)
#     return a * b

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

@mcp.tool()
async def evaluate_responses():
    """
    Evaluate model responses e.g Retrieval, Groundedness, Relevance, Response Completeness etc.
    
    Args:
        promtp: Prompt for the evaluation.

    Returns:
        str: Evaluation result in string.
    """
    env_path = os.path.dirname(os.path.dirname(__file__)) + os.path.sep

    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=openaiendpoint,
        api_key=openaikey,
        azure_deployment=openapideploymentname,
        api_version=aiapiversion,
    )

    # Initializing Groundedness and Groundedness Pro evaluators
    groundedness_eval = GroundednessEvaluator(model_config)
    relevance_eval = RelevanceEvaluator(model_config)
    retrieval_eval = RetrievalEvaluator(model_config)

    result = evaluate(
        data=env_path + "/data/evaluation_data.jsonl", # provide your data here
        evaluators={
            "groundedness": groundedness_eval,
            "relevance": relevance_eval,
            "retrieval": retrieval_eval

        },
        # column mapping
        evaluator_config={
            "default": {
                "column_mapping": {
                    "query": "${data.query}",
                    "context": "${data.context}",
                    "response": "${data.response}"
                } 
            }
        }
    )

    evaluation_result = str(json.dumps(result, indent=4))
    return evaluation_result

@mcp.tool()
async def execute_quantum_job(repetitions_count):
    """
    Execute a quantum process using QPU (Quantum Processing Unit).
    
    Args:
        repetitions_count: Number of repetitions for the QPU process.

    Returns:
        json: QPU process result.
    """
     
    workspace = Workspace.from_connection_string(azure_quantum_conn_str)
    service = AzureQuantumService(workspace)

    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(
        cirq.H(q0),               # Apply an H-gate to q0
        cirq.measure(q0)          # Measure q0
    )
    circuit

     # To view the probabilities computed for each Qubit state, you can print the result.

    result = service.run(
        program=circuit,
        repetitions=repetitions_count,
        target="ionq.simulator",
        timeout_seconds=500 # Set timeout to accommodate queue time on QPU
    )
    
    resultStr = str(result)
    resultStr = resultStr.split('=')[1]
    charzero = resultStr.count("0")
    charone = resultStr.count("1")

    quantum_response = {
        "result": result,
        "zero": charzero,
        "one": charone,
        "circuit": str(circuit)
    }
    return quantum_response

if __name__ == "__main__":
    print("Starting Math Server")
    mcp.run(transport="stdio")