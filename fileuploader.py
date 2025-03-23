import os
import uuid
from pathlib import Path
from typing import Optional, Union, BinaryIO
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, ContentSettings
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AzureBlobStorage:
    """Client for interacting with Azure Blob Storage"""
    
    def __init__(self, connection_string: Optional[str] = None, container_name: str = "miscdocs"):
        """
        Initialize the Azure Blob Storage client
        
        Args:
            connection_string: Azure Storage connection string
            container_name: Name of the container to use
        """
        # Load environment variables
        env_path = Path('.') / 'secrets.env'
        load_dotenv(dotenv_path=env_path)
        
        # Get connection string from environment if not provided
        self.connection_string = os.environ.get("AZURE_BLOB_CONN_STR")
        if not self.connection_string:
            raise ValueError("Azure Storage connection string is required")
            
        self.container_name = container_name
        
        # Initialize the client
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        
        # Ensure container exists
        self._ensure_container_exists()
        
    def _ensure_container_exists(self) -> None:
        """Ensure that the specified container exists"""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            container_client.get_container_properties()
            logger.info(f"Container '{self.container_name}' already exists")
        except ResourceNotFoundError:
            logger.info(f"Creating container '{self.container_name}'")
            self.blob_service_client.create_container(self.container_name)
            
    def upload_blob(self, 
                   data: Union[bytes, str, BinaryIO], 
                   blob_name: Optional[str] = None,
                   content_type: Optional[str] = None) -> str:
        """
        Upload binary data to Azure Blob Storage
        
        Args:
            data: Binary data to upload (bytes, file-like object, or string path)
            blob_name: Name for the blob (generated if not provided)
            content_type: MIME type of the data
            
        Returns:
            URL to the uploaded blob
        """
        # Generate a unique blob name if not provided
        if not blob_name:
            blob_name = f"{uuid.uuid4()}"
            
            # Add file extension based on content type if available
            if content_type:
                if content_type == "image/jpeg":
                    blob_name += ".jpg"
                elif content_type == "image/png":
                    blob_name += ".png"
                elif content_type == "application/pdf":
                    blob_name += ".pdf"
                    
        # Get a blob client
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )
        
        # Set content settings if provided
        content_settings = None
        if content_type:
            content_settings = ContentSettings(content_type=content_type)
        
        # Upload the data
        try:
            logger.info(f"Uploading blob '{blob_name}' to container '{self.container_name}'")
            
            # Handle different input types
            if isinstance(data, bytes):
                blob_client.upload_blob(data, overwrite=True, content_settings=content_settings)
            elif isinstance(data, str) and os.path.isfile(data):
                # It's a file path
                with open(data, "rb") as file:
                    blob_client.upload_blob(file, overwrite=True, content_settings=content_settings)
            else:
                # Assume it's a file-like object
                blob_client.upload_blob(data, overwrite=True, content_settings=content_settings)
                
            # Get the blob URL
            blob_url = blob_client.url
            logger.info(f"Blob uploaded successfully: {blob_url}")
            return blob_url
            
        except Exception as e:
            logger.error(f"Error uploading blob: {e}")
            raise
            
    def download_blob(self, blob_name: str) -> bytes:
        """
        Download a blob from Azure Storage
        
        Args:
            blob_name: Name of the blob to download
            
        Returns:
            Binary data of the blob
        """
        try:
            # Get a blob client
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            
            # Download the blob
            download_stream = blob_client.download_blob()
            return download_stream.readall()
            
        except ResourceNotFoundError:
            logger.error(f"Blob '{blob_name}' not found")
            raise
        except Exception as e:
            logger.error(f"Error downloading blob: {e}")
            raise


def upload_file_to_azure(file_path: str, container_name: str = "default") -> str:
    """
    Convenience function to upload a file to Azure Blob Storage
    
    Args:
        file_path: Path to the file to upload
        container_name: Name of the container
        
    Returns:
        URL to the uploaded file
    """
    client = AzureBlobStorage(container_name=container_name)
    
    # Determine content type based on file extension
    content_type = None
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.jpg' or file_extension == '.jpeg':
        content_type = "image/jpeg"
    elif file_extension == '.png':
        content_type = "image/png"
    elif file_extension == '.pdf':
        content_type = "application/pdf"
    elif file_extension == '.txt':
        content_type = "text/plain"
    elif file_extension == '.html':
        content_type = "text/html"
    
    # Use the file name as the blob name
    blob_name = os.path.basename(file_path)
    
    return client.upload_blob(file_path, blob_name=blob_name, content_type=content_type)
    
def upload_binary_to_azure(data: bytes, blob_name: Optional[str] = None, 
                          content_type: Optional[str] = None,
                          container_name: str = "default") -> str:
    """
    Convenience function to upload binary data to Azure Blob Storage
    
    Args:
        data: Binary data to upload
        blob_name: Name for the blob (generated if not provided)
        content_type: MIME type of the data
        container_name: Name of the container
        
    Returns:
        URL to the uploaded binary data
    """
    client = AzureBlobStorage(container_name=container_name)
    return client.upload_blob(data, blob_name=blob_name, content_type=content_type)


# Example usage
if __name__ == "__main__":
    # Example 1: Upload a file
    try:
        # Replace with an actual file path
        file_path = "C:/Users/cchakraborty/Downloads/IMG_1175.jpg"
        file_url = upload_file_to_azure(file_path, container_name="miscdocs")
        print(f"File uploaded successfully: {file_url}")
    except Exception as e:
        print(f"Error uploading file: {e}")
        
    # Example 2: Upload binary data
    # try:
    #     binary_data = b"Hello, Azure Blob Storage!"
    #     binary_url = upload_binary_to_azure(
    #         binary_data, 
    #         blob_name="hello.txt",
    #         content_type="text/plain", 
    #         container_name="documents"
    #     )
    #     print(f"Binary data uploaded successfully: {binary_url}")
    # except Exception as e:
    #     print(f"Error uploading binary data: {e}")