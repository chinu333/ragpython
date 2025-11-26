# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Update package lists and install Graphviz + build tooling for pygraphviz
RUN apt-get update && apt-get install -y graphviz graphviz-dev libgraphviz-dev pkg-config build-essential unixodbc unixodbc-dev \
	&& rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# RUN pip install pygraphviz --global-option=build_ext --global-option="--include-path=C:\Program Files\Graphviz\include" --global-option="--library-path=C:\Program Files\Graphviz\lib"
RUN pip install --no-cache-dir pygraphviz

# Install any dependencies specified in requirements.txt
RUN pip install --use-deprecated=legacy-resolver --no-cache-dir --prefer-binary -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Make port 443 available to the world outside this container
EXPOSE 443

# Run copilot_agents.py when the container launches to run as streamlit app
CMD ["streamlit", "run", "copilot_agents.py"]

# Run agentsapi.py when the container launches to run as api
# CMD ["fastapi", "run", "agentsapi.py", "--port", "80"]

# Run a2aserver when the container launches
# CMD ["python", "a2aserver", "--host", "0.0.0.0", "--port", "80"]