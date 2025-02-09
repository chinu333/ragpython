from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import *
from azure.search.documents.indexes.models import (
    SimpleField,
    SearchFieldDataType,
    SearchableField,
    SearchField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
    SearchIndex
)

from pathlib import Path  
import os
from dotenv import load_dotenv
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


if __name__ == "__main__":

    env_path = Path('.') / 'secrets.env'
    load_dotenv(dotenv_path=env_path)

    aisearchindexname = os.getenv("AZURE_AI_SEARCH_INDEX_NAME")
    aisearchkey = os.getenv("AZURE_AI_SEARCH_KEY")
    openaikey = os.getenv("AZURE_OPENAI_API_KEY")
    openaiendpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    aisearchendpoint= os.getenv("AZURE_AI_SEARCH_SERVICE_ENDPOINT")
    search_creds = AzureKeyCredential(aisearchkey)
    embeddingname = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
    aiapiversion = os.getenv("AZURE_OPENAI_API_VERSION")

    azureaiclient = AzureOpenAI(
        api_key=openaikey,  
        api_version=aiapiversion,
        azure_endpoint = openaiendpoint
    )


print(aisearchindexname)

# Option 2: Use AzureOpenAIEmbeddings with an Azure account
embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    azure_deployment=embeddingname,
    openai_api_version=aiapiversion,
    azure_endpoint=openaiendpoint,
    api_key=openaikey,
)

# Specify additional properties for the Azure client such as the following https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core/README.md#configurations
vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=aisearchendpoint,
    azure_search_key=aisearchkey,
    index_name=aisearchindexname,
    embedding_function=embeddings.embed_query,
    # Configure max retries for the Azure client
    additional_search_client_options={"retry_total": 4},
)

# loader = TextLoader("./data/ms10k_2024.txt")
loader = PyPDFLoader("./data/Azure_StackHub.pdf")

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
vector_store.add_documents(documents=docs)