from langgraph_cua import create_cua
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


cua_graph = create_cua(scrapybara_api_key="scrapy-75640ce4-c331-4a63-a03a-233745b96fc7")

# Define the input messages
messages = [
    {
        "role": "system",
        "content": (
            "You're an advanced AI computer use assistant. The browser you are using "
            "is already initialized, and visiting google.com."
        ),
    },
    {
        "role": "user",
        "content": (
            "Can you find the best price for new all season tires which will fit on my 2019 Subaru Forester?"
        ),
    },
]

async def main():
    # Stream the graph execution
    stream = cua_graph.astream(
        {"messages": messages},
        stream_mode="updates"
    )

    # Process the stream updates
    async for update in stream:
        if "create_vm_instance" in update:
            print("VM instance created")
            stream_url = update.get("create_vm_instance", {}).get("stream_url")
            # Open this URL in your browser to view the CUA stream
            print(f"Stream URL: {stream_url}")

    print("Done")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())