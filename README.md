# Agentic Framework Implementation

This repository demonstrates the implementation of the Agentic Framework, showcasing twelve distinct agent capabilities. The agents included are:

- **RAG Agent**
- **SQL Agent**
- **Weather Agent**
- **Financial Advisor Agent**
- **Generic Agent**
- **Multimodality Agent**
- **Visualization Agent**
- **Mermaid Agent**
- **Email Agent**
- **Traffic Agent**
- **Graph RAG Agent**
- **Image Generation Agent**

## Tech Stack

The following technologies are utilized in this repository:

- **Python**: The primary programming language used for developing the agents.
- **LangChain**: A framework for building applications with language models.
- **LangGraph**: A tool for creating and managing language model workflows.
- **LangSmith**: LangSmith is an all-in-one developer platform to debug, collaborate, test, and monitor your LLM applications.
- **Azure OpenAI GPT Model**: Used for generating natural language responses.
- **Azure OpenAI Embedding Model**: Used for embedding and similarity search tasks.
- **Azure AI Search**: Used for storing embedded data and performing advanced RAG, search and retrieval.

## Agents Description

### RAG Agent
- **Purpose**: Retrieve and generate information based on user queries.
- **Capabilities**: Combines retrieval-augmented generation to provide accurate and contextually relevant answers.

### SQL Agent
- **Purpose**: Interact with SQL databases to execute queries and manage data. We are using 'northwind' db for this demo and the db is included under data folder
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

### Visualization Agent
- **Purpose**: Visulalize structured data in pie, bar and scatter charts.
- **Capabilities**: Integrates different types of data to provide comprehensive visualization.

### Mermaid Agent
- **Purpose**: Draw architecture diagram in mermaid format (Diagram as Code) from text based user input.
- **Capabilities**: Converts user descriptions into visual diagrams using mermaid code, making it easier to understand complex systems and workflows.

### Email Agent
- **Purpose**: Send emails to specified recipients with a given subject and message.
- **Capabilities**: Automates the process of sending emails, ensuring timely and accurate communication with the intended recipients.

### Traffic Agent
- **Purpose**: Provide traffic updates between two specified locations.
- **Capabilities**: Analyzes traffic data in JSON format to offer real-time traffic conditions, helping users plan their routes more efficiently.

### Graph RAG Agent
- **Purpose**: Retrieve answers from a graph database based on user questions.
- **Capabilities**: Utilizes graph databases to provide precise and relevant answers, leveraging the relationships and connections within the data.

### Image Generation Agent
- **Purpose**: Generate images based on user questions.
- **Capabilities**: Utilizes DALL-E-3 model to generate images based on the user prompt.

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
6. Run the ```python ingestion.py``` to vectorize and store `ms10k_2024.txt`file (or you can bring your own file)
7. Run the ```streamlit run copilot_agents.py``` to run

## Contributing

We welcome contributions to enhance the capabilities of the Agentic Framework. Please follow the standard GitHub workflow for contributing:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.