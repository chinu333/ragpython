from azure.search.documents.indexes.models import *
from pathlib import Path  
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import json
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain.agents.agent_types import AgentType 
from tenacity import retry, wait_random_exponential, stop_after_attempt  
import pandas as pd
from langchain.prompts.chat import ChatPromptTemplate



env_path = Path('.') / 'secrets.env'
load_dotenv(dotenv_path=env_path)

openaikey = os.getenv("AZURE_OPENAI_API_KEY")
openaiendpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openapideploymentname = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT_NAME")
aiapiversion = os.getenv("AZURE_OPENAI_API_VERSION")

sqllite_db_path= os.environ.get("SQLITE_DB_PATH","data/northwind.db")
engine = create_engine(f'sqlite:///{sqllite_db_path}')

def execute_sql_query(sql_query, limit=100):  
    result = pd.read_sql_query(sql_query, engine)
    result = result.infer_objects()
    for col in result.columns:  
        if 'date' in col.lower():  
            result[col] = pd.to_datetime(result[col], errors="ignore")  

    # result = result.head(limit)  # limit to save memory  
    # st.write(result)
    # print(result)
    return result

llm = AzureChatOpenAI(
    azure_deployment=openapideploymentname,
    azure_endpoint=openaiendpoint,
    openai_api_key=openaikey,
    api_version=aiapiversion,
    verbose=False,
    temperature=0,
)

def generate_sql(question):

    # Load the metadata file
    with open(os.getenv("META_DATA_FILE","data/metadata.json"), "r") as file:
        data = json.load(file)

    messages = [
        (
            "system",
            "You are a helpful assistant that generates SQL query for northwind db based on the user question. The context is the northwind database schema and the info mentioned here: {data}",
            # "You are a helpful assistant that generates SQL query for northwind db based on the user question. The context is the northwind database schema",
        ),
        ("human", question),
    ]
    response = llm.invoke(messages)
    # print("Question :: " + question)
    # print("AI Aisstant :: " + response.content + "\n")
    return response.content

def execute_sql(sql):
    return execute_sql_query(sql)

def generate_response_from_sql(question):
        
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
            "You are a helpful assistant that generates SQL query for northwind db based on the user question. The context is the northwind database schema. Limit your reponses with 10 rows.",
            ),
            ("user", "{question}\n AI: "),
        ]
    )

    db = SQLDatabase(engine)
    sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    sql_toolkit.get_tools()

    sqldb_agent = create_sql_agent(
        llm=llm,
        toolkit=sql_toolkit,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=False
    )

    response = sqldb_agent.invoke(prompt.format(
        question=question
    ))
    
    return response['output']

# print(execute_sql("select * from [order details] limit 5;"))

# while True:
#     user_input = input("\nUser Question (or 'q' to quit): ")

#     if user_input == 'q':
#         break

#     try:
#         question = str(user_input)
#         print(generate_response_from_sql(question))
#     except ValueError:
#         print("Invalid input. Please enter a number or 'q'.")