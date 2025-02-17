from langchain_community.graphs import Neo4jGraph
from pathlib import Path  
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.chains import GraphCypherQAChain

env_path = Path('.') / 'secrets.env'
load_dotenv(dotenv_path=env_path)

neo4juri = os.getenv("NEO4J_URI")
neo4juser = os.getenv("NEO4J_USERNAME")
neo4jpassword = os.getenv("NEO4J_PASSWORD")
openaikey = os.getenv("AZURE_OPENAI_API_KEY")
openaiendpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openapideploymentname = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT_NAME")
aiapiversion = os.getenv("AZURE_OPENAI_API_VERSION")


# llm = AzureChatOpenAI(
#         azure_deployment=openapideploymentname,
#         azure_endpoint=openaiendpoint,
#         openai_api_key=openaikey,
#         api_version=aiapiversion,
#         verbose=False,
#         temperature=0,
#     )

def get_llm():
    llm = AzureChatOpenAI(
        azure_deployment=openapideploymentname,
        azure_endpoint=openaiendpoint,
        openai_api_key=openaikey,
        api_version=aiapiversion,
        verbose=False,
        temperature=0,
    )
    return llm

# graph=Neo4jGraph(
#     url=neo4juri,
#     username=neo4juser,
#     password=neo4jpassword,
# )

def connect_graph():
    graph=Neo4jGraph(
        url=neo4juri,
        username=neo4juser,
        password=neo4jpassword,
    )
    return graph

### Load the dataset of movie

movie_query="""
LOAD CSV WITH HEADERS FROM
'https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/movies/movies_small.csv' as row

MERGE(m:Movie{id:row.movieId})
SET m.released = date(row.released),
    m.title = row.title,
    m.imdbRating = toFloat(row.imdbRating)
FOREACH (director in split(row.director, '|') |
    MERGE (p:Person {name:trim(director)})
    MERGE (p)-[:DIRECTED]->(m))
FOREACH (actor in split(row.actors, '|') |
    MERGE (p:Person {name:trim(actor)})
    MERGE (p)-[:ACTED_IN]->(m))
FOREACH (genre in split(row.genres, '|') |
    MERGE (g:Genre {name:trim(genre)})
    MERGE (m)-[:IN_GENRE]->(g))
"""

def populate_graph(graph):
    # graph.query(movie_query)
    # graph.refresh_schema()
    print(graph.schema)

def populate_graph_with_docs(graph):
    # loader = TextLoader("./data/ms10k_2024.txt")
    # documents = loader.load()
    # llm_transformer=LLMGraphTransformer(llm=llm)
    # graph_documents=llm_transformer.convert_to_graph_documents(documents)
    # # graph.refresh_schema()
    # print("Graph Documents Relationship :: " + graph_documents[0].relationships)
     print(graph.schema)

def ask_graph_db(question):
    chain=GraphCypherQAChain.from_llm(llm=get_llm(), graph=connect_graph(), verbose=True, allow_dangerous_requests=True)
    response=chain.invoke({question})
    print("Graph RAG Response :: " + response["result"])
    # graph.close()
    return response["result"]

# ask_graph_db("Identify 10 movies where directors also played a role in the film.")