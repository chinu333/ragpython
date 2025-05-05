from langsmith import Client
from langsmith import evaluate
from pathlib import Path  
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import uuid

env_path = Path('.') / 'secrets.env'
load_dotenv(dotenv_path=env_path)

openaiendpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
openaikey = os.getenv("AZURE_OPENAI_API_KEY")
openapideploymentname = os.getenv("AZURE_OPENAI_GPT4_DEPLOYMENT_NAME")
aiapiversion = os.getenv("AZURE_OPENAI_API_VERSION")

judge_llm = AzureChatOpenAI(
    azure_deployment=openapideploymentname,
    azure_endpoint=openaiendpoint,
    openai_api_key=openaikey,
    api_version=aiapiversion,
    verbose=False,
    temperature=0,
)

questions = [
    "What was Microsoft\'s cloud revenue for 2024?",
    "Did linkedin's revenue grow in 2024?"
]
answers = [
    "Microsoft's cloud revenue for 2024 was $137.4 billion.",
    "Yes, LinkedIn's revenue grew in 2024."
]

ls_client = Client()
ls_client.delete_dataset(dataset_name="Agent Evaluation")  # Delete the dataset if it exists

dataset_name = "Agent Evaluation"
example_inputs = [
   ("RAG: What was Microsoft\'s cloud revenue for 2024?", "Microsoft's cloud revenue for 2024 was $137.4 billion."),
   ("RAG: Did linkedin's revenue grow in 2024?", "Yes, LinkedIn's revenue grew in 2024."),
]

dataset = ls_client.create_dataset(
   dataset_name=dataset_name,
   description="Dataset for evaluating the performance of the RAG agent.",
)

print("Dataset ID:: ", dataset.id)
print("Dataset Name:: ", dataset.name)

ls_client.create_examples(
   dataset_id=dataset.id,
   inputs=[{"question": q} for q, _ in example_inputs],
   outputs=[{"answer": a} for _, a in example_inputs],   
)

def get_thread_id():
    """
    Function to get the thread ID.
    
    Returns:
        str: The thread ID.
    """
    # Generate thread id if the chat history is empty
    return str(uuid.uuid4())


config = {
    "configurable": {
        "thread_id": get_thread_id()
    }
}

def correct(outputs: dict, reference_outputs: dict) -> bool:
    instructions = (
        "Given an actual answer and an expected answer, determine whether"
        " the actual answer contains all of the information in the"
        " expected answer. Respond with 'CORRECT' if the actual answer"
        " does contain all of the expected information and 'INCORRECT'"
        " otherwise. Do not include anything else in your response."
    )
    # Our graph outputs a State dictionary, which in this case means
    # we'll have a 'messages' key and the final message should
    # be our actual answer.
    actual_answer = outputs["messages"][-1].content
    expected_answer = reference_outputs["answer"]
    user_msg = (
        f"ACTUAL ANSWER: {actual_answer}"
        f"\n\nEXPECTED ANSWER: {expected_answer}"
    )
    response = judge_llm.invoke(
        [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_msg}
        ]
    )
    return response.content.upper() == "CORRECT"

def example_to_state(inputs: dict) -> dict:
  return {"messages": [{"role": "user", "content": inputs['question']}]}

def evaluate_agents(app):
    # We use LCEL declarative syntax here.
    # Remember that langgraph graphs are also langchain runnables.
    app.config = config
    target = example_to_state | app

    experiment_results = evaluate(
        target,
        data=dataset_name,
        evaluators=[correct],
        max_concurrency=4,  # optional
        experiment_prefix="Agentic-AI-Demo-Baseline",  # optional
    )