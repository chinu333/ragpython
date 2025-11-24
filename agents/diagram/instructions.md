# Agent Instructions: Azure Architecture Diagram Generation

## Diagram Generation Workflow

### Complete Process (1 Step)

#### Step 1: Create Python Diagram Script
- Import required Azure components from `diagrams.azure.*`
- Use proper icon names (e.g., `PublicIpAddresses` not `PublicIPAddresses`)
- Configure graph attributes for layout:
  ```python
  graph_attr = {
      "splines": "ortho",      # Orthogonal lines
      "nodesep": "0.8",        # Node spacing
      "ranksep": "1.2",        # Rank spacing
      "fontsize": "12",
      "bgcolor": "white",
      "pad": "0.5"
  }
  ```
- Use Cluster for logical grouping (VNets, Subnets, Resource Groups)
- Set different background colors for different tiers/clusters
- Set output format: `outformat=["png"]`
- Set output filename: `filename="./images/architecture_diagram`

---

## Color Coding for Tiers

Use different background colors to distinguish architectural tiers:

```python
# Frontend Tier
frontend_cluster_attr = {
    "fontsize": "12",
    "bgcolor": "#E3F2FD",  # Light Blue
    "style": "rounded",
    "margin": "15"
}

# Database Tier
database_cluster_attr = {
    "fontsize": "12",
    "bgcolor": "#FFF3E0",  # Light Orange
    "style": "rounded",
    "margin": "15"
}

# Load Balancer
lb_cluster_attr = {
    "fontsize": "12",
    "bgcolor": "#F3E5F5",  # Light Purple
    "style": "rounded",
    "margin": "15"
}

# Availability Set
avset_cluster_attr = {
    "fontsize": "12",
    "bgcolor": "#E8F5E9",  # Light Green
    "style": "rounded",
    "margin": "15"
}
```

Apply to clusters:
```python
with Cluster("Frontend Subnet", graph_attr=frontend_cluster_attr):
    # components
```

---

## Common Azure Icon Imports

```python
from diagrams import Diagram, Cluster, Edge

# Compute
from diagrams.azure.compute import VM, AvailabilitySets, FunctionApps, ContainerInstances

# Network
from diagrams.azure.network import (
    VirtualNetworks, Subnets, LoadBalancers, 
    ApplicationGateway, FrontDoors, 
    NetworkSecurityGroupsClassic, 
    PublicIpAddresses, NetworkInterfaces,
    PrivateEndpoint, DNSPrivateZones
)

# Database
from diagrams.azure.database import SQLServers, SQLDatabases

# Storage
from diagrams.azure.storage import StorageAccounts, BlobStorage

# Security
from diagrams.azure.security import KeyVaults

# Identity
from diagrams.azure.identity import ManagedIdentities

# Integration
from diagrams.azure.integration import ServiceBus

# Monitoring
from diagrams.azure.analytics import LogAnalyticsWorkspaces
from diagrams.azure.devops import ApplicationInsights
```

**Important**: Always check actual available class names using:
```python
from diagrams.azure import network
print([x for x in dir(network) if not x.startswith('_')])
```

---

## Troubleshooting

### Import Errors
**Error**: `cannot import name 'PublicIPAddresses'`

**Solution**: Check exact class name (case-sensitive):
- Correct: `PublicIpAddresses`
- Wrong: `PublicIPAddresses`

### Cluttered Layout
**Issue**: Auto-generated diagrams have messy layouts

**Solutions**:
1. Adjust graph attributes (`nodesep`, `ranksep`)
2. Use `direction="TB"` or `"LR"`
3. Simplify cluster nesting
4. **Best approach**: Auto-generate, then manually refine in draw.io

---

## Key Principles

1. **Auto-generate first, refine later**: Use Python scripts to get the diagram
3. **Color code tiers**: Makes diagrams easier to understand at a glance
4. **Use clusters liberally**: Group related resources (VNets, Subnets, Resource Groups)
5. **Edge labels**: Label connections with protocols/ports for clarity

---