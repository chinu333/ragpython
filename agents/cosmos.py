from azure.cosmos import CosmosClient
import uuid
import logging
from pathlib import Path  
import os
from dotenv import load_dotenv

env_path = os.path.dirname(os.path.dirname(__file__)) + os.path.sep + 'secrets.env'
load_dotenv(dotenv_path=env_path)

cosmosuri = os.getenv("COSMOS_URI")
cosmoskey = os.getenv("COSMOS_KEY")
cosmosdbname = os.getenv("COSMOS_DB_NAME")
cosmoscontainername = os.getenv("COSMOS_CONTAINER_NAME")


def save_to_cosmos(prompt, response):
    print("Saving to Cosmos DB...")
    print("Prompt: ", prompt)
    print("Response: ", response)
    logging.info("Prompt: %s", prompt)
    logging.info("Response: %s", response)

    if prompt and response:
        URL = cosmosuri
        KEY = cosmoskey
        client = CosmosClient(URL, credential=KEY)

        DATABASE_NAME = cosmosdbname
        database = client.get_database_client(DATABASE_NAME)

        CONTAINER_NAME = cosmoscontainername
        container = database.get_container_client(CONTAINER_NAME)

        container.upsert_item({
                'id': str(uuid.uuid4()),
                'info': "prompt :: " + prompt + " || response :: " + response,
            }
        )
    else:
        logging.error("Prompt or response is empty. Cannot save to Cosmos DB.")
        return "Prompt or response is empty."

    return "Data saved to Cosmos DB successfully."

# save_to_cosmos("test", "testresponse")