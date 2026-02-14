"""
Edge Deployment Manager

Orchestrates end-to-end deployment of models to edge devices.
Handles packaging, distribution, installation, and lifecycle management.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import time


class DeploymentStatus(Enum):
    """Status of deployment."""
    PREPARING = "preparing"
    PACKAGING = "packaging"
    UPLOADING = "uploading"
    INSTALLING = "installing"
    VALIDATING = "validating"
    ACTIVE = "active"
    INACTIVE = "inactive"
    FAILED = "failed"


@dataclass
class DeploymentPackage:
    """Package for deployment to edge."""
    package_id: str
    model_name: str
    model_version: str
    target_device: str
    
    model_file_path: str
    weights_file_path: str
    quantization_config_path: Optional[str] = None
    
    package_size_mb: float = 0.0
    compression_format: str = "zip"
    checksum: str = ""
    
    created_timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeploymentRecord:
    """Record of deployment event."""
    deployment_id: str
    package_id: str
    target_device: str
    device_id: str
    
    deployment_status: DeploymentStatus
    deployment_timestamp: float
    completion_timestamp: Optional[float] = None
    
    latency_seconds: float = 0.0
    success: bool = False
    error_message: Optional[str] = None
    
    pre_deployment_metrics: Dict[str, float] = field(default_factory=dict)
    post_deployment_metrics: Dict[str, float] = field(default_factory=dict)


class EdgeDeploymentManager:
    """Manages model deployment to edge devices."""
    
    def __init__(self):
        self.deployments: Dict[str, DeploymentRecord] = {}
        self.packages: Dict[str, DeploymentPackage] = {}
        self.active_models: Dict[str, Dict] = {}
    
    def create_deployment_package(
        self,
        model_name: str,
        model_version: str,
        model_path: str,
        target_device: str,
        optimizations: Optional[Dict] = None,
    ) -> DeploymentPackage:
        """Create deployment package."""
        import uuid
        
        package_id = f"pkg_{model_name}_{model_version}_{uuid.uuid4().hex[:8]}"
        
        # Simulate package creation
        package = DeploymentPackage(
            package_id=package_id,
            model_name=model_name,
            model_version=model_version,
            target_device=target_device,
            model_file_path=model_path,
            weights_file_path=f"{model_path}.weights",
            package_size_mb=100.0,
        )
        
        if optimizations:
            if "quantization" in optimizations:
                package.quantization_config_path = "quantization.json"
        
        self.packages[package_id] = package
        return package
    
    def deploy_to_device(
        self,
        package: DeploymentPackage,
        device_id: str,
        deployment_fn: Optional[callable] = None,
    ) -> DeploymentRecord:
        """Deploy package to device."""
        import uuid
        
        deployment_id = f"dep_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        deployment_record = DeploymentRecord(
            deployment_id=deployment_id,
            package_id=package.package_id,
            target_device=package.target_device,
            device_id=device_id,
            deployment_status=DeploymentStatus.PREPARING,
            deployment_timestamp=start_time,
        )
        
        try:
            # Step 1: Validate package
            deployment_record.deployment_status = DeploymentStatus.VALIDATING
            if not self._validate_package(package):
                raise Exception("Package validation failed")
            
            # Step 2: Upload to device
            deployment_record.deployment_status = DeploymentStatus.UPLOADING
            if deployment_fn:
                deployment_fn(device_id, package)
            else:
                self._simulate_upload(device_id, package)
            
            # Step 3: Install
            deployment_record.deployment_status = DeploymentStatus.INSTALLING
            self._simulate_installation(package, device_id)
            
            # Step 4: Validate deployment
            deployment_record.deployment_status = DeploymentStatus.VALIDATING
            
            # Success
            deployment_record.deployment_status = DeploymentStatus.ACTIVE
            deployment_record.success = True
            deployment_record.completion_timestamp = time.time()
            deployment_record.latency_seconds = deployment_record.completion_timestamp - start_time
            
            # Record active model
            self.active_models[device_id] = {
                "model": package.model_name,
                "version": package.model_version,
                "deployment_id": deployment_id,
            }
        
        except Exception as e:
            deployment_record.deployment_status = DeploymentStatus.FAILED
            deployment_record.error_message = str(e)
            deployment_record.completion_timestamp = time.time()
            deployment_record.latency_seconds = deployment_record.completion_timestamp - start_time
        
        self.deployments[deployment_id] = deployment_record
        return deployment_record
    
    def _validate_package(self, package: DeploymentPackage) -> bool:
        """Validate deployment package."""
        if not package.model_file_path:
            return False
        if package.package_size_mb <= 0:
            return False
        return True
    
    def _simulate_upload(self, device_id: str, package: DeploymentPackage) -> None:
        """Simulate uploading package to device."""
        # Simulate upload latency
        time.sleep(0.1)
    
    def _simulate_installation(self, package: DeploymentPackage, device_id: str) -> None:
        """Simulate installation on device."""
        # Simulate installation latency
        time.sleep(0.05)
    
    def get_active_model(self, device_id: str) -> Optional[Dict]:
        """Get currently active model on device."""
        return self.active_models.get(device_id)
    
    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentRecord]:
        """Get status of deployment."""
        return self.deployments.get(deployment_id)
    
    def list_deployments(self, device_id: Optional[str] = None) -> List[DeploymentRecord]:
        """List deployments optionally filtered by device."""
        if device_id:
            return [d for d in self.deployments.values() if d.device_id == device_id]
        return list(self.deployments.values())
    
    def rollback_deployment(self, deployment_id: str) -> bool:
        """Rollback to previous deployment."""
        if deployment_id not in self.deployments:
            return False
        
        record = self.deployments[deployment_id]
        record.deployment_status = DeploymentStatus.INACTIVE
        return True
    
    def get_deployment_statistics(self) -> Dict[str, Any]:
        """Get deployment statistics."""
        total = len(self.deployments)
        successful = sum(1 for d in self.deployments.values() if d.success)
        failed = total - successful
        
        avg_latency = 0.0
        if successful > 0:
            avg_latency = sum(d.latency_seconds for d in self.deployments.values() if d.success) / successful
        
        return {
            "total_deployments": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0,
            "average_latency_seconds": avg_latency,
            "active_devices": len(self.active_models),
        }
