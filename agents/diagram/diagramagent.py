from azure.search.documents.indexes.models import *
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI


env_path = os.path.dirname(os.path.dirname(os.path.dirname( __file__ ))) + os.path.sep + 'secrets.env'
# print("Env Path :: " + env_path)
load_dotenv(dotenv_path=env_path)

openaiendpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openaikey = os.getenv("AZURE_OPENAI_API_KEY")
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

def read_instruction():
    """
    Reads the content of a Markdown file.

    Args:
        no args

    Returns:
        str: The content of the Markdown file as a string.
             Returns an empty string if the file is not found or an error occurs.
    """
    try:
        with open("./agents/diagram/instructions.md", 'r', encoding='utf-8') as f:
        # with open("./instructions.md", 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        print(f"Error: The file was not found.")
        return ""
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return ""

def read_azureclassnames():
    """
    Reads the content of a Markdown file.

    Args:
        no args

    Returns:
        str: The content of the Markdown file as a string.
             Returns an empty string if the file is not found or an error occurs.
    """
    try:
        with open("./agents/diagram/azureclassnames.md", 'r', encoding='utf-8') as f:
        # with open("./azureclassnames.md", 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        print(f"Error: The file was not found.")
        return ""
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return ""

def draw(prompt):
    # print("Prompt :: " + prompt)

    instructions = read_instruction()
    # print("Instruction :: " + instructions)

    azureclassnames = read_azureclassnames()
    # print("Azure Class Names :: " + azureclassnames)

    sysMsg = "Read the instructions as follows:" + instructions + " and generate python script to draw architecture diagram. Import required Azure components from `diagrams.azure.*`. Before importing any package or class name, please make sure it appears in the azure class names documentation. Classnames are mentioned here:\n" + azureclassnames + "\nPlease follow case sensitivity of the class and package names. Always import `FunctionApps` class from `diagrams.azure.compute` package. Respond only with the code and nothing else. Do not include any explanations or instructions and any python code marker like ```python or ```."
    # print("System Message :: " + sysMsg)

    messages = [
        (
            "system",
            sysMsg
        ),
        ("human", prompt),
    ]
    response = llm.invoke(messages)
    # print("Question :: " + question)
    # print("AI Aisstant :: " + response.content + "\n")
    generate_diagram(response.content)
    return "./architecture_diagram.png"

def generate_diagram(code: str):
    if not code.strip():
        print("No code returned to execute.")
        return
    try:
        exec(code)
    except Exception as exc:
        print(f"Error executing generated diagram code: {exc}")


# draw("Generate a event driven, resilient architecture diagram in Azure Cloud based on these resources: App Service, Function App, Storage Account, Redis Cache, SQL Database, Cosmos DB, Azure CDN, Azure Front Door, Azure DNS. Spread the workload in two different regions, active and passive and route the traffic from single Front Door, global load balancer to each region accordingly. The flow should start from Azure DNS and then to Azure Front Door and then to the regions. The traffic must flow to the regions from Front Door. Azure CDN must be connected to the Azure Front Door and CDN should take the static content from Azure Storage. In each region App Service receives the traffic from Front Door then traffic goes to Function App and then to the appropriate data sources.")