from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
import base64
from pathlib import Path  
import os
from dotenv import load_dotenv


env_path = os.path.dirname(os.path.dirname( __file__ )) + os.path.sep + 'secrets.env'
load_dotenv(dotenv_path=env_path)

docintel_endpoint = os.getenv("DOC_INTEL_ENDPOINT")
docintel_key = os.getenv("DOC_INTEL_KEY")


def extract_text_from_doc(file_name):
    with open("./data/" + file_name, "rb") as f:
        base64_encoded_pdf = base64.b64encode(f.read()).decode("utf-8")

    analyze_request = {
        "base64Source": base64_encoded_pdf
    }

    document_intelligence_client = DocumentIntelligenceClient(
        endpoint=docintel_endpoint, credential=AzureKeyCredential(docintel_key)
    )

    poller = document_intelligence_client.begin_analyze_document(
        "prebuilt-layout", analyze_request
    )

    result = poller.result()

    if result.styles and any([style.is_handwritten for style in result.styles]):
        print("Document contains handwritten content")
    else:
        print("Document does not contain handwritten content")

    docs = []

    for page in result.pages:
        print(f"----Analyzing layout from page #{page.page_number}----")
        print(
            f"Page has width: {page.width} and height: {page.height}, measured with unit: {page.unit}"
        )

        if page.lines:
            for line in page.lines:
                # print(f"Content: {line.content}")
                docs.append(line.content)  # Print content of the line
    return docs
# Call the function to analyze the layout of the locally downloaded file analyze_layout_local_file("YOUR_FULL_LOCAL_PATH_TO_PDF_FILE")
# print(extract_text_from_doc("D050_AK_2023.pdf"))