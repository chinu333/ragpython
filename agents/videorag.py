
from azure.search.documents.indexes.models import *
from pathlib import Path  
import os
from dotenv import load_dotenv
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.schema import Document
import uuid
import json
from agents.cuclient import AzureContentUnderstandingClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser



env_path = os.path.dirname(os.path.dirname( __file__ )) + os.path.sep + 'secrets.env'
load_dotenv(dotenv_path=env_path)

aisearchindexname = os.getenv("VIDEO_INDEX_NAME")
aisearchkey = os.getenv("AZURE_AI_SEARCH_KEY")
aisearchendpoint= os.getenv("AZURE_AI_SEARCH_SERVICE_ENDPOINT")

embeddingkey = os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY")
embeddingendpoint = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
embeddingname = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

openaiendpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openaikey = os.getenv("AZURE_OPENAI_API_KEY")
openapideploymentname = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT_NAME")
aiapiversion = os.getenv("AZURE_OPENAI_API_VERSION")

azure_cu_endpoint = os.getenv("AZURE_CONTENT_UNDERSTANDING_ENDPOINT")
azure_cu_api_version = os.getenv("AZURE_CONTENT_UNDERSTANDING_API_VERSION")
azure_cu_key = os.getenv("AZURE_CONTENT_UNDERSTANDING_KEY")
azure_cu_index = os.getenv("VIDEO_INDEX_NAME")

embeddings = AzureOpenAIEmbeddings(
    model=embeddingname,
    azure_endpoint=embeddingendpoint,
    openai_api_key=embeddingkey,
)

# vector_store: AzureSearch = AzureSearch(
#     embedding_function=embeddings.embed_query,
#     azure_search_endpoint=aisearchendpoint,
#     azure_search_key=aisearchkey,
#     index_name=aisearchindexname,
# )

llm = AzureChatOpenAI(
    azure_deployment=openapideploymentname,
    azure_endpoint=openaiendpoint,
    openai_api_key=openaikey,
    api_version=aiapiversion,
    verbose=False,
    temperature=0,
)

vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=aisearchendpoint,
    azure_search_key=aisearchkey,
    index_name=aisearchindexname,
    embedding_function=embeddings.embed_query
)

credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")

def convert_values_to_strings(json_obj):
    return [str(value) for value in json_obj]


def remove_markdown(json_obj):
    for segment in json_obj:
        if 'markdown' in segment:
            del segment['markdown']
    return json_obj


def process_cu_scene_description(scene_description):
    audio_visual_segments = scene_description["result"]["contents"]
    filtered_audio_visual_segments = remove_markdown(audio_visual_segments)
    audio_visual_splits = [
        "The following is a json string representing a video segment with scene description and transcript ```"
        + v
        + "```"
        for v in convert_values_to_strings(filtered_audio_visual_segments)
    ]
    docs = [Document(page_content=v) for v in audio_visual_splits]
    return docs

def embed_and_index_chunks(docs):
    # embeddings = AzureOpenAIEmbeddings(
    #     model=embeddingname,
    #     azure_endpoint=embeddingendpoint,
    #     openai_api_key=embeddingkey,
    # )

    vector_store.add_documents(documents=docs)
    print("Video documents ingested into the vector store.")
    return vector_store


def ingest_vector_data():
    """
    Ingests data from a file into the vector store.

    """
    VIDEO_LOCATION = Path("../data/CopilotStudioDelivery.mp4")
    ANALYZER_TEMPLATE_PATH = Path("../data/video_content_understanding.json")
    ANALYZER_ID = "video_analyzer" + "_" + str(uuid.uuid4())  # Unique identifier for the analyzer

    # Create the Content Understanding (CU) client
    cu_client = AzureContentUnderstandingClient(
        endpoint=azure_cu_endpoint,
        api_version=azure_cu_api_version,
        subscription_key=azure_cu_key,
        token_provider=token_provider,
        x_ms_useragent="azure-ai-content-understanding-python/search_with_video", # This header is used for sample usage telemetry, please comment out this line if you want to opt out.
    )

    # Use the client to create an analyzer
    response = cu_client.begin_create_analyzer(ANALYZER_ID, analyzer_template_path=ANALYZER_TEMPLATE_PATH)
    result = cu_client.poll_result(response)

    print(json.dumps(result, indent=2))

    # Submit the video for content analysis
    response = cu_client.begin_analyze(ANALYZER_ID, file_location=VIDEO_LOCATION)

    # Wait for the analysis to complete and get the content analysis result
    video_cu_result = cu_client.poll_result(response, timeout_seconds=3600)  # 1 hour timeout

    # Print the content analysis result
    print(f"Video Content Understanding result: ", video_cu_result)

    # Delete the analyzer if it is no longer needed
    cu_client.delete_analyzer(ANALYZER_ID)
    docs = process_cu_scene_description(video_cu_result)
    print("There are " + str(len(docs)) + " documents.")
    # embed and index the docs:
    embed_and_index_chunks(docs)

def ask_video_rag(question):
    """
    Ask a question to the video RAG system.

    Args:
        question: The question to ask.

    Returns:
        The answer to the question.
    """
    # response = ask_vector_store(question)
    # return response
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
        # for doc in docs:
        #     # source = doc.metadata["source"]
        #     # source = source.replace("./data/", "")
        #     print("Source: ", doc)
        #     sources.append(doc)
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
    print("Question :: " + question)
    print("AI Aisstant :: " + result + "\n")
    return result

# ingest_vector_data()
# print(ask_video_rag("What was request received for the pushback?"))
