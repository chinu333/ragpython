from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes.models import *
from pathlib import Path  
import os
from dotenv import load_dotenv
from langchain_community.vectorstores.azuresearch import AzureSearch
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
openaikey = os.getenv("AZURE_OPENAI_API_KEY")
openaiendpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
aisearchendpoint= os.getenv("AZURE_AI_SEARCH_SERVICE_ENDPOINT")
search_creds = AzureKeyCredential(aisearchkey)
embeddingname = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
openapideploymentname = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT_NAME")
aiapiversion = os.getenv("AZURE_OPENAI_API_VERSION")

embeddings = AzureOpenAIEmbeddings(
    model=embeddingname,
    azure_endpoint=openaiendpoint,
    openai_api_key=openaikey,
)

vector_store: AzureSearch = AzureSearch(
    embedding_function=embeddings.embed_query,
    azure_search_endpoint=aisearchendpoint,
    azure_search_key=aisearchkey,
    index_name=aisearchindexname,
)

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
        for doc in docs:
            source = doc.metadata["source"]
            source = source.replace("./data/", "")
            sources.append(source)
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {
            "context": vector_store.as_retriever() | format_docs,
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