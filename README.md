# Introduction

# Agentic Framework Implementation

This repository demonstrates the implementation of the Agentic Framework, showcasing six distinct agent capabilities. The agents included are:

- **RAG Agent**
- **SQL Agent**
- **Weather Agent**
- **Financial Advisor Agent**
- **Generic Agent**
- **Multimodality Agent**

## Tech Stack

The following technologies are utilized in this repository:

- **Python**: The primary programming language used for developing the agents.
- **LangChain**: A framework for building applications with language models.
- **LangGraph**: A tool for creating and managing language model workflows.
- **Azure OpenAI GPT Model**: Used for generating natural language responses.
- **Azure OpenAI Embedding Model**: Used for embedding and similarity search tasks.

## Agents Description

### RAG Agent
- **Purpose**: Retrieve and generate information based on user queries.
- **Capabilities**: Combines retrieval-augmented generation to provide accurate and contextually relevant answers.

### SQL Agent
- **Purpose**: Interact with SQL databases to execute queries and manage data.
- **Capabilities**: Can perform complex queries, and data analysis tasks.

### Weather Agent
- **Purpose**: Provide weather updates and forecasts.
- **Capabilities**: Fetches real-time weather data and forecasts based on user location or specified parameters.

### Financial Advisor Agent
- **Purpose**: Offer financial advice and insights.
- **Capabilities**: Analyzes financial data, provides investment recommendations, and answers finance-related queries.

### Generic Agent
- **Purpose**: Serve as a versatile agent for various tasks.
- **Capabilities**: Can be customized to handle a wide range of tasks based on user requirements.

### Multimodality Agent
- **Purpose**: Handle tasks involving multiple modes of input and output (e.g., text, images).
- **Capabilities**: Integrates different types of data to provide comprehensive responses and solutions.

# Getting Started
Guide users through getting your code up and running on their own system. In this section you can talk about:
1.	Installation process
2.	Software dependencies
3.	Latest releases
4.	API references

# How to run
1. Clone the repo (e.g. ```git clone https://github.com/chinu333/ragpython.git``` or download). Then navigate to ```cd ragpython```
2. Provide settings for Azure Open AI and Database in a file named `secrets.env` file in the root of this folder.
4. Create a python environment with version from 3.8 and 3.10
    - [Python 3+](https://www.python.org/downloads/)
        - **Important**: Python and the pip package manager must be in the path in Windows for the setup scripts to work.
        - **Important**: Ensure you can run `python --version` from console. On Ubuntu, you might need to run `sudo apt install python-is-python3` to link `python` to `python3`. 
5. Import the requirements.txt `pip install -r requirements.txt`
6. Add your question(s) in 'questions' variable (line 261)
7. Run the ```python copilot_agents.py``` to run