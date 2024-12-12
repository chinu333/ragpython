
from pathlib import Path  
import os 
from sqlalchemy import create_engine 
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain.agents.agent_types import AgentType
import streamlit as st 
from langchain.callbacks.base import BaseCallbackHandler


env_path = Path('.') / 'secrets.env'
load_dotenv(dotenv_path=env_path)

openaikey = os.getenv("AZURE_OPENAI_API_KEY")
openaiendpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openapideploymentname = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT_NAME")
aiapiversion = os.getenv("AZURE_OPENAI_API_VERSION")

class SQLHandler(BaseCallbackHandler):
    def __init__(self):
        self.sql_result = None

    def on_agent_action(self, action, **kwargs):
        """Run on agent action. if the tool being used is sql_db_query,
         it means we're submitting the sql and we can 
         record it as the final sql"""

        if action.tool == "sql_db_query":
            self.sql_result = action.tool_input

llm = AzureChatOpenAI(
    azure_deployment=openapideploymentname,
    azure_endpoint=openaiendpoint,
    openai_api_key=openaikey,
    api_version=aiapiversion,
    verbose=False,
    temperature=0,
)

sqllite_db_path= os.environ.get("SQLITE_DB_PATH","data/northwind.db")
engine = create_engine(f'sqlite:///{sqllite_db_path}')

today = pd.Timestamp.today()
#format today's date
today = today.strftime("%Y-%m-%d")

def get_dataframe(question):

    CODER1 = f"""
    You are a helpful assistant that generates SQL query for northwind db based on the user question. 
    Today's date is {today}. The data is housed in a SQLITE database. The context is the northwind database schema.
    You will review the relevant business rules and table schemas that pertain to the user's query to adeptly craft your steps for answering their questions.
    If the query is intricate, employ best practices in business analytics to decompose it into manageable steps, articulating your analytical process to the user throughout. Conclude by presenting your findings in a clear, succinct manner, employing visualizations when beneficial to optimally convey your insights.

    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
            CODER1
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
        handle_parsing_errors=True,
        verbose=False
    )

    handler = SQLHandler()

    response = sqldb_agent.invoke(prompt.format(
        question=question
    ), {"callbacks": [handler]})

    print(response['output'])

    sql = str(handler.sql_result)
    sql = sql[11:len(sql) - 2]
    sql = sql.replace("\\n", " ")
    print("SQL :: ", sql)


    result = pd.read_sql_query(sql, engine)
    result = result.infer_objects()
    return response['output'], result 