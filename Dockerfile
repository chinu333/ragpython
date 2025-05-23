# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any dependencies specified in requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Make port 443 available to the world outside this container
EXPOSE 443

# Run copilot_agents.py when the container launches to run as streamlit app
CMD ["streamlit", "run", "copilot_agents.py"]

# Run agentsapi.py when the container launches to run as api
# CMD ["fastapi", "run", "agentsapi.py", "--port", "80"]