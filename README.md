# Agentic Framework Implementation

This repository demonstrates the implementation of the Agentic Framework, showcasing fifteen distinct agent capabilities. The agents included are:

- **RAG Agent**
- **SQL Agent**
- **Nutrition Agent**
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
- **Developer Agent**
- **Search Agent**

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

### Nutrition Agent
- **Purpose**: Provides nutrition info (e.g calorie, protein, fat, carb etc) of food items
- **Capabilities**: Makes an api call to nutrionix.com to get the nutrition info and return the info in text format.

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

### Developer Agent
- **Purpose**: Generate code based on user questions.
- **Capabilities**: Utilizes o1-mini model to generate code based on the user prompt.

### Search Agent
- **Purpose**: Search web for any queries.
- **Capabilities**: Utilizes Bing/Google search capabilities to search on the web based on the user prompt.

# Getting Started
Guide users through getting your code up and running on their own system. In this section you can talk about:
1.	Installation process
2.	Software dependencies
3.	Latest releases
4.	API references

# How to run
1. Clone the repo (e.g. ```git clone https://github.com/chinu333/ragpython.git``` or download). Then navigate to ```cd ragpython```
2. Provide settings for Azure Open AI and Database in a file named `secrets.env` file in the root of this folder.
3. Make sure you have installed Python 3.10
    - [Python 3.10](https://www.python.org/downloads/)
        - **Important**: Python and the pip package manager must be in the path in Windows for the setup scripts to work.
        - **Important**: Ensure you can run `python --version` from console. On Ubuntu, you might need to run `sudo apt install python-is-python3` to link `python` to `python3`. 
4. Import the requirements.txt `pip install -r requirements.txt`
5. Run the ```python ingestion.py``` to vectorize and store `ms10k_2024.txt`file (or you can bring your own file)
6. Run the ```streamlit run copilot_agents.py``` to run

# Sameple Questions
1. RAG: What was Microsoft\'s cloud revenue for 2024? (RAG Agent)
2. Compare Lucid and Tesla stocks and provide a recommendation for which one to buy. (Financial Advisor Agent)
3. Tell me something about Quantum Computing. (Generic Agent)
4. What are the total sales broken down by country? Show in a pie chart. (SQL Agent & Visualization Agent)
5. Please give me month-wise break up of the quantity of 'Chai' sold throughout the year of 2016. (SQL Agent)
6. Analyze the architecture diagram image and generate Terraform code for deploying all the resources in Azure. Please put all the resource in one resource group and the use the name rg_agents for the resource group. Image URL: https://ragstorageatl.blob.core.windows.net/miscdocs/WAF.png (Multimodality Agent)
7. What kind of cloth I need to wear today? I am in Atlanta, GA. Please also suggest a couple of stores in Atlanta where I can buy the clothes. (Weather Agent & Generic/Search Agent)
8. GRAPH RAG: Give me the list of 10 movies having imdbRating more than 8? (Graph RAG Agent)
9. Generate sequence diagram in mermaid js for Airport Management System. (Mermaid Agent)
10. What time I need to start from 200 17th St NW, Atlanta, GA to reach Hartsfield-Jackson Atlanta Airport at 7PM? Please provide the time breakdown between delay and actual travel time. Please mention the distance in miles unit. (Traffic Agent)
11. Generate a image of panoramic view of the Grand Canyon at sunrise, layers of rock bathed in vibrant hues of orange, red, and purple, the Colorado River snaking through the depths. (Image Generation Agent)
12. SEARCH: What are the top restaurants near Microsoft Atlanta office? Please provide their website link as well. What is the acronym for DOGE in the context of US Government. (Search Agent)
13. DEVELOPER: Write self contained code in HTML and JavaScript for scientific calculator. (Developer Agent)
14. NUTRITION: Provide nutrition info of 1 cup starbucks coffee venti white chocolate mocha with whipped cream (Nutrition Agent)
15. PHI: what is the capital of France? (PHI Agent)
16. DEEPSEEK: Write self contained code in HTML and JavaScript for scientific calculator. (DeepSeek Agent)
17. SPACE: Show me the NASA picture of the day today? (Space Agent)
18. SPACE: Get the mars weather information. Compare the mars weather with Atlanta, GA (Space & Weather Agent)


## Contributing

We welcome contributions to enhance the capabilities of the Agentic Framework. Please follow the standard GitHub workflow for contributing:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.