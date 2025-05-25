import json
import os
from pathlib import Path
from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.evaluation import GroundednessEvaluator, AzureOpenAIModelConfiguration, evaluate, QAEvaluator, RelevanceEvaluator, RetrievalEvaluator


env_path = Path('.') / 'secrets.env'
load_dotenv(dotenv_path=env_path)

openaiendpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openaikey = os.getenv("AZURE_OPENAI_API_KEY")
openapideploymentname = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT_NAME")
aiapiversion = os.getenv("AZURE_OPENAI_API_VERSION")


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
qa_eval = QAEvaluator(model_config=model_config, threshold=3)

# ai_project = {
#     "subscription_id": "dfbce8c6-7e40-4c00-b0c2-7542f1dd2814",
#     "resource_group_name": "rg-cchakrabortyai",
#     "project_name": "mtcatlcc",
# }

result = evaluate(
    data="./data/evaluation_data.jsonl", # provide your data here
    evaluators={
        "groundedness": groundedness_eval,
        "relevance": relevance_eval,
        "retrieval": retrieval_eval,
        # "qa": qa_eval

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
    },
    # azure_ai_project=ai_project
)

print("Evaluation Result: ", json.dumps(result, indent=4))