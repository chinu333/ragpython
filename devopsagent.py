from mcp.server.fastmcp import FastMCP
import requests
import subprocess

# Create an MCP server
mcp = FastMCP("DevOpsAgent")

@mcp.tool()
def get_pypi_package_info(package_name: str) -> dict:
    """
    Fetch package info from PyPI API.

    Args:
        package_name (str): The name of the PyPI package.

    Returns:
        dict: Dictionary with maintainers, versions, description, and extra info.
    """
    url = f"https://pypi.org/pypi/{package_name}/json"
    resp = requests.get(url)
    if resp.status_code != 200:
        return {"error": f"Package '{package_name}' not found on PyPI."}

    data = resp.json()
    info = data.get("info", {})
    releases = data.get("releases", {})

    maintainers = []
    if info.get("maintainer"):
        maintainers.append(info["maintainer"])
    if info.get("author"):
        maintainers.append(info["author"])

    result = {
        "name": info.get("name"),
        "summary": info.get("summary"),
        "description": info.get("description"),
        "maintainers": list(set(maintainers)),
        "latest_version": info.get("version"),
        "all_versions": sorted(releases.keys(), reverse=True),
        "home_page": info.get("home_page"),
        "license": info.get("license"),
        "project_url": info.get("project_url"),
        "package_url": info.get("package_url"),
        "requires_python": info.get("requires_python"),
        "keywords": info.get("keywords"),
    }
    return result

@mcp.tool()
def build_docker_image(image_name: str, version: str, dockerfile_path: str = "Dockerfile", context_path: str = ".") -> dict:
    """
    Build a Docker image using the specified name and version.

    Args:
        image_name (str): The name of the Docker image.
        version (str): The version/tag for the Docker image.
        dockerfile_path (str): Path to the Dockerfile (default: "Dockerfile").
        context_path (str): Build context directory (default: current directory).

    Returns:
        dict: Result of the build process.
    """
    tag = f"{image_name}:{version}"
    cmd = [
        "docker", "build",
        "-t", tag,
        "-f", dockerfile_path,
        context_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return {
            "success": True,
            "image_tag": tag,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "image_tag": tag,
            "stdout": e.stdout,
            "stderr": e.stderr,
            "error": str(e)
        }

@mcp.tool()
def push_docker_image_to_acr(
    image_name: str,
    version: str,
    acr_login_server: str,
    acr_username: str,
    acr_password: str
) -> dict:
    """
    Push a Docker image to Azure Container Registry (ACR).

    Args:
        image_name (str): The local Docker image name.
        version (str): The image version/tag.
        acr_login_server (str): The ACR login server (e.g., myregistry.azurecr.io).
        acr_username (str): The ACR username.
        acr_password (str): The ACR password.

    Returns:
        dict: Result of the push process.
    """
    local_tag = f"{image_name}:{version}"
    acr_tag = f"{acr_login_server}/{image_name}:{version}"

    # Tag the image for ACR
    tag_cmd = [
        "docker", "tag", local_tag, acr_tag
    ]
    # Login to ACR
    login_cmd = [
        "docker", "login", acr_login_server,
        "-u", acr_username,
        "-p", acr_password
    ]
    # Push the image
    push_cmd = [
        "docker", "push", acr_tag
    ]

    try:
        tag_result = subprocess.run(tag_cmd, capture_output=True, text=True, check=True)
        login_result = subprocess.run(login_cmd, capture_output=True, text=True, check=True)
        push_result = subprocess.run(push_cmd, capture_output=True, text=True, check=True)
        return {
            "success": True,
            "acr_image_tag": acr_tag,
            "tag_stdout": tag_result.stdout,
            "login_stdout": login_result.stdout,
            "push_stdout": push_result.stdout,
            "push_stderr": push_result.stderr
        }
    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "acr_image_tag": acr_tag,
            "stdout": e.stdout,
            "stderr": e.stderr,
            "error": str(e)
        }

@mcp.tool()
def deploy_to_azure_container_app(
    resource_group: str,
    container_app_name: str,
    acr_image_tag: str,
    acr_login_server: str,
    acr_username: str,
    acr_password: str,
    location: str = "eastus"
) -> dict:
    """
    Deploy a Docker image from Azure Container Registry to Azure Container App.

    Args:
        resource_group (str): Azure resource group name.
        container_app_name (str): Name for the Azure Container App.
        acr_image_tag (str): Full image tag (e.g., myregistry.azurecr.io/myimage:tag).
        acr_login_server (str): ACR login server (e.g., myregistry.azurecr.io).
        acr_username (str): ACR username.
        acr_password (str): ACR password.
        location (str): Azure region (default: eastus).

    Returns:
        dict: Result of the deployment process.
    """
    import subprocess

    # Ensure Azure CLI is logged in and extension is installed
    try:
        # Create resource group if not exists
        subprocess.run([
            "az", "group", "create",
            "--name", resource_group,
            "--location", location
        ], capture_output=True, text=True, check=True)

        # Create Container App environment if not exists
        env_name = f"{container_app_name}-env"
        subprocess.run([
            "az", "containerapp", "env", "create",
            "--name", env_name,
            "--resource-group", resource_group,
            "--location", location
        ], capture_output=True, text=True, check=True)

        # Create or update the Container App
        deploy_cmd = [
            "az", "containerapp", "create",
            "--name", container_app_name,
            "--resource-group", resource_group,
            "--environment", env_name,
            "--image", acr_image_tag,
            "--registry-server", acr_login_server,
            "--registry-username", acr_username,
            "--registry-password", acr_password,
            "--ingress", "external",
            "--target-port", "80"
        ]
        result = subprocess.run(deploy_cmd, capture_output=True, text=True, check=True)
        return {
            "success": True,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "stdout": e.stdout,
            "stderr": e.stderr,
            "error": str(e)
        }

if __name__ == "__main__":
    mcp.run(transport="sse")