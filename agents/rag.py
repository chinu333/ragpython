from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes.models import *
from pathlib import Path  
import os
from dotenv import load_dotenv
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_community.vectorstores.azure_cosmos_db_no_sql import AzureCosmosDBNoSqlVectorSearch
from langchain_community.vectorstores.azure_cosmos_db_no_sql import CosmosDBQueryType
from azure.cosmos import CosmosClient, PartitionKey
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# if __name__ == "__main__":

env_path = os.path.dirname(os.path.dirname( __file__ )) + os.path.sep + 'secrets.env'
load_dotenv(dotenv_path=env_path)

aisearchindexname = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")
aisearchkey = os.getenv("AZURE_AI_SEARCH_KEY")

embeddingkey = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY")
embeddingendpoint = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")

aisearchendpoint= os.getenv("AZURE_AI_SEARCH_SERVICE_ENDPOINT")
search_creds = AzureKeyCredential(aisearchkey)
embeddingname = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
embeddingapiversion = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION")

openaiendpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openaikey = os.getenv("AZURE_OPENAI_API_KEY")
openapideploymentname = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT_NAME")
aiapiversion = os.getenv("AZURE_OPENAI_API_VERSION")

cosmosuri = os.getenv("COSMOS_URI")
cosmoskey = os.getenv("COSMOS_KEY")
cosmosdbname = os.getenv("COSMOS_DB_NAME")
cosmoscontainername = os.getenv("COSMOS_CONTAINER_NAME")

embeddings = AzureOpenAIEmbeddings(
    model=embeddingname,
    azure_endpoint=embeddingendpoint,
    openai_api_key=embeddingkey,
)

# Instantiate AzureOpenAIEmbeddings
# embeddings = AzureOpenAIEmbeddings(
#     model="text-embedding-3-large",  # Or your specific embedding model
#     azure_deployment=embeddingname,
#     openai_api_version=embeddingapiversion,
#     azure_endpoint=embeddingendpoint,
#     openai_api_key=embeddingkey
# )

vector_store: AzureSearch = AzureSearch(
    embedding_function=embeddings.embed_query,
    azure_search_endpoint=aisearchendpoint,
    azure_search_key=aisearchkey,
    index_name=aisearchindexname,
)

# def get_cosmosdb_vector_store():
#     indexing_policy = {
#         "indexingMode": "consistent",
#         "includedPaths": [{"path": "/*"}],
#         "excludedPaths": [{"path": '/"_etag"/?'}],
#         "vectorIndexes": [{"path": "/embedding", "type": "diskANN"}],
#         "fullTextIndexes": [{"path": "/text"}],
#     }

#     vector_embedding_policy = {
#         "vectorEmbeddings": [
#             {
#                 "path": "/embedding",
#                 "dataType": "float32",
#                 "distanceFunction": "cosine",
#                 "dimensions": 1536,
#             }
#         ]
#     }

#     partition_key = PartitionKey(path="/id")
#     cosmos_container_properties = {"partition_key": partition_key}

#     cosmosclient = CosmosClient(cosmosuri, credential=cosmoskey)

#     # insert the documents in AzureCosmosDBNoSql with their embedding
#     cosmos_vector_store = AzureCosmosDBNoSqlVectorSearch(
#         embedding=embeddings,
#         cosmos_client=cosmosclient,
#         database_name=cosmosdbname,
#         container_name="cosmosvectorstore",
#         vector_embedding_policy=vector_embedding_policy,
#         indexing_policy=indexing_policy,
#         cosmos_container_properties=cosmos_container_properties,
#         cosmos_database_properties={}
#     )

#     return cosmos_vector_store

llm = AzureChatOpenAI(
    azure_deployment=openapideploymentname,
    azure_endpoint=openaiendpoint,
    openai_api_key=openaikey,
    api_version=aiapiversion,
    verbose=False,
    temperature=0,
)

def ask_vector_store(question):

    PROMPT_TEMPLATE = """You are an AI Assistant. Given the following context:
    {context}

    Answer the following question:
    {question}

    Assistant:"""

    PROMPT = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )

    sources = []

    def format_docs(docs):
        # Optionally trim to top 5 after retrieving 50
        trimmed = docs[:len(docs) - 1]
        # for doc in docs:
        for doc in trimmed:
            source = doc.metadata["source"]
            source = source.replace("./data/", "")
            sources.append(source)
        # return "\n\n".join(doc.page_content for doc in docs)
        return "\n\n".join(doc.page_content for doc in trimmed)

    qa_chain = (
        {
            "context": vector_store.as_retriever(
                search_type="hybrid",
                k=5  # broaden recall; format_docs limits to 5
                # k=50
            ) | format_docs,
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    

    # question = "What was Microsoft's cloud revenue for 2024?"
    # question = "You are the secretary of the VP of Microsoft Azure Cloud. Write am email to the shareholders reporting Microsoft's cloud revenue from 2020 to 2024."
    response = qa_chain.invoke(question)
    result = response
    # print("Question :: " + question)
    # print("AI Aisstant :: " + result + "\n")
    distinct_sources = []
    for source in sources:
        if source not in distinct_sources:
            distinct_sources.append(source)
    print("Sources :: ", distinct_sources)
    
    result_with_sources = result + """

Sources:

    """
    for source in distinct_sources:
        result_with_sources += f" ** {source}" + "\n"

    # print("Result with sources :: ")
    # print(result_with_sources)
    return result_with_sources

# print(ask_vector_store("What was Microsoft's cloud revenue for 2024?"))