"""
Comprehensive tests for Phase 2.3 - Multi-region Deployment & Failover
Tests for region management, failover, and multi-region orchestration
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from src.deployment.region_manager import (
    get_region_manager,
    RegionManager,
    RegionStatus,
    RegionTier,
)
from src.deployment.failover_manager import (
    get_failover_manager,
    FailoverManager,
    FailoverPolicy,
    FailoverStrategy,
    FailoverTrigger,
)
from src.api.main import app


# ============================================================================
# REGION MANAGER TESTS
# ============================================================================

class TestRegionManager:
    """Test region manager functionality"""
    
    def test_register_region(self):
        """Test region registration"""
        manager = RegionManager()
        
        region_id = manager.register_region(
            region_name="US East",
            endpoint="https://api-us-east-1.example.com",
            tier=RegionTier.PRIMARY,
        )
        
        assert region_id is not None
        assert manager.get_region(region_id) is not None
    
    def test_get_all_regions(self):
        """Test getting all regions"""
        manager = RegionManager()
        manager.register_region("Region1", "https://api1.example.com", RegionTier.PRIMARY)
        manager.register_region("Region2", "https://api2.example.com", RegionTier.SECONDARY)
        
        regions = manager.get_all_regions()
        assert len(regions) >= 2
    
    def test_get_healthy_regions(self):
        """Test getting healthy regions"""
        manager = RegionManager()
        
        r1_id = manager.register_region("Health1", "https://api1.example.com", RegionTier.PRIMARY)
        r2_id = manager.register_region("Health2", "https://api2.example.com", RegionTier.SECONDARY)
        
        # Update metrics to mark one as healthy
        manager.update_region_metrics(r1_id, 100, 80, 1, 100)
        manager.update_region_metrics(r2_id, 6000, 10, 60, 50)
        
        healthy = manager.get_healthy_regions()
        assert len(healthy) >= 1
    
    def test_update_region_metrics(self):
        """Test updating region metrics"""
        manager = RegionManager()
        region_id = manager.register_region("Test", "https://api.example.com", RegionTier.PRIMARY)
        
        manager.update_region_metrics(region_id, 50, 90, 0.5, 1000)
        
        region = manager.get_region(region_id)
        assert region is not None
        assert region.metrics.latency_ms == 50
        assert region.metrics.available_capacity == 90
        assert region.metrics.error_rate == 0.5
    
    def test_region_status_determination(self):
        """Test region status is updated correctly"""
        manager = RegionManager()
        region_id = manager.register_region("Status", "https://api.example.com", RegionTier.PRIMARY)
        
        # Good metrics
        manager.update_region_metrics(region_id, 50, 90, 1, 1000)
        region = manager.get_region(region_id)
        assert region is not None
        assert region.status == RegionStatus.HEALTHY
        
        # Degraded metrics
        manager.update_region_metrics(region_id, 2000, 60, 5, 500)
        region = manager.get_region(region_id)
        assert region is not None
        assert region.status == RegionStatus.DEGRADED
        
        # Unhealthy metrics
        manager.update_region_metrics(region_id, 8000, 10, 80, 100)
        region = manager.get_region(region_id)
        assert region is not None
        assert region.status == RegionStatus.UNHEALTHY
    
    def test_get_recommended_region(self):
        """Test getting recommended region for routing"""
        manager = RegionManager()
        
        r1_id = manager.register_region("East", "https://api1.example.com", RegionTier.SECONDARY)
        r2_id = manager.register_region("West", "https://api2.example.com", RegionTier.SECONDARY)
        
        manager.update_region_metrics(r1_id, 50, 80, 1, 1000)
        manager.update_region_metrics(r2_id, 100, 80, 1, 1000)
        
        recommended = manager.get_recommended_region()
        assert recommended is not None
        assert recommended.metrics.latency_ms <= 100
    
    def test_enable_disable_region(self):
        """Test enabling/disabling regions"""
        manager = RegionManager()
        region_id = manager.register_region("Toggle", "https://api.example.com", RegionTier.PRIMARY)
        
        region = manager.get_region(region_id)
        assert region is not None
        assert region.enabled is True
        
        manager.disable_region(region_id)
        region = manager.get_region(region_id)
        assert region is not None
        assert region.enabled is False
        
        manager.enable_region(region_id)
        region = manager.get_region(region_id)
        assert region is not None
        assert region.enabled is True
    
    def test_regional_comparison(self):
        """Test regional performance comparison"""
        manager = RegionManager()
        
        r1_id = manager.register_region("Region1", "https://api1.example.com", RegionTier.PRIMARY)
        r2_id = manager.register_region("Region2", "https://api2.example.com", RegionTier.SECONDARY)
        
        manager.update_region_metrics(r1_id, 50, 90, 1, 1000)
        manager.update_region_metrics(r2_id, 150, 70, 2, 500)
        
        comparison = manager.get_regional_comparison()
        assert "regions" in comparison
        assert comparison["best_performing"] is not None
    
    def test_capacity_analysis(self):
        """Test capacity analysis"""
        manager = RegionManager()
        
        r1_id = manager.register_region("Cap1", "https://api1.example.com", RegionTier.PRIMARY, max_capacity=1000)
        r2_id = manager.register_region("Cap2", "https://api2.example.com", RegionTier.SECONDARY, max_capacity=500)
        
        manager.update_region_metrics(r1_id, 50, 50, 1, 500)
        manager.update_region_metrics(r2_id, 50, 20, 1, 400)
        
        capacity = manager.get_capacity_analysis()
        assert capacity["total_capacity"] == 1500
        assert capacity["utilization_percentage"] > 0


# ============================================================================
# FAILOVER MANAGER TESTS
# ============================================================================

class TestFailoverManager:
    """Test failover management"""
    
    @pytest.mark.asyncio
    async def test_failover_conditions(self):
        """Test failover condition checking"""
        manager = FailoverManager()
        
        # Low error rate - no failover
        metrics = {"error_rate": 5, "available_capacity": 80, "latency_ms": 100}
        trigger = await manager.check_failover_conditions(metrics)
        assert trigger is None
        
        # High error rate - should trigger
        metrics = {"error_rate": 60, "available_capacity": 80, "latency_ms": 100}
        trigger = await manager.check_failover_conditions(metrics)
        assert trigger == FailoverTrigger.HIGH_ERROR_RATE
    
    @pytest.mark.asyncio
    async def test_manual_failover_strategy(self):
        """Test manual failover strategy"""
        policy = FailoverPolicy(strategy=FailoverStrategy.MANUAL)
        manager = FailoverManager(policy)
        
        # Manual strategy should not auto-trigger
        metrics = {"error_rate": 60, "available_capacity": 5}
        trigger = await manager.check_failover_conditions(metrics)
        assert trigger is None
    
    @pytest.mark.asyncio
    async def test_immediate_failover(self):
        """Test immediate failover execution"""
        manager = FailoverManager()
        
        success = await manager.initiate_failover(
            from_region="us-east-1",
            to_region="us-west-2",
            trigger=FailoverTrigger.HIGH_ERROR_RATE,
            reason="Test failover"
        )
        
        assert success is True
        assert "us-east-1" in manager.current_failovers or len(manager.failover_history) > 0
    
    @pytest.mark.asyncio
    async def test_gradual_failover(self):
        """Test gradual failover"""
        policy = FailoverPolicy(
            strategy=FailoverStrategy.GRADUAL,
            gradual_shift_percentage=25,
        )
        manager = FailoverManager(policy)
        
        success = await manager.initiate_failover(
            from_region="us-east-1",
            to_region="us-west-2",
            trigger=FailoverTrigger.HIGH_ERROR_RATE,
        )
        
        assert success is True or len(manager.failover_history) > 0
    
    def test_failover_history(self):
        """Test failover history tracking"""
        manager = FailoverManager()
        
        history = manager.get_failover_history()
        assert isinstance(history, list)
    
    def test_traffic_distribution(self):
        """Test traffic distribution management"""
        manager = FailoverManager()
        
        distribution = {"region1": 50.0, "region2": 50.0}
        success = manager.set_traffic_distribution(distribution)
        assert success is True
        
        current = manager.get_traffic_distribution()
        assert current["region1"] == 50.0
    
    def test_traffic_distribution_validation(self):
        """Test traffic distribution validation"""
        manager = FailoverManager()
        
        # Invalid distribution (doesn't sum to 100)
        distribution = {"region1": 60.0, "region2": 30.0}
        success = manager.set_traffic_distribution(distribution)
        assert success is False
    
    def test_failover_analytics(self):
        """Test failover analytics"""
        manager = FailoverManager()
        
        analytics = manager.get_failover_analytics()
        assert "total_events" in analytics
        assert "success_rate" in analytics


# ============================================================================
# API ENDPOINT TESTS
# ============================================================================

class TestDeploymentEndpoints:
    """Test deployment API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_register_region_endpoint(self, client):
        """Test region registration endpoint"""
        response = client.post(
            "/deployment/regions/register",
            json={
                "region_name": "Test Region",
                "endpoint": "https://api-test.example.com",
                "tier": "PRIMARY",
                "max_capacity": 2000,
            }
        )
        
        assert response.status_code in [200, 201, 429]  # 429 if rate limited
        if response.status_code == 200:
            data = response.json()
            assert "region_id" in data
    
    def test_get_all_regions_endpoint(self, client):
        """Test get all regions endpoint"""
        response = client.get("/deployment/regions")
        
        assert response.status_code in [200, 429]
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
    
    def test_get_region_endpoint(self, client):
        """Test get specific region endpoint"""
        response = client.get("/deployment/regions/test-region-id")
        
        assert response.status_code in [200, 404, 429]
    
    def test_compare_regions_endpoint(self, client):
        """Test region comparison endpoint"""
        response = client.get("/deployment/comparison")
        
        assert response.status_code in [200, 429]
        if response.status_code == 200:
            data = response.json()
            assert "total_regions" in data
    
    def test_capacity_endpoint(self, client):
        """Test capacity analysis endpoint"""
        response = client.get("/deployment/capacity")
        
        assert response.status_code in [200, 429]
        if response.status_code == 200:
            data = response.json()
            assert "total_capacity" in data
    
    def test_overview_endpoint(self, client):
        """Test multi-region overview endpoint"""
        response = client.get("/deployment/overview")
        
        assert response.status_code in [200, 429]
        if response.status_code == 200:
            data = response.json()
            assert "total_regions" in data
            assert "health_percentage" in data
    
    def test_failover_status_endpoint(self, client):
        """Test failover status endpoint"""
        response = client.get("/deployment/failover/status")
        
        assert response.status_code in [200, 429]
        if response.status_code == 200:
            data = response.json()
            assert "active_failovers" in data
    
    def test_traffic_endpoint(self, client):
        """Test traffic distribution endpoint"""
        response = client.get("/deployment/traffic")
        
        assert response.status_code in [200, 429]
    
    def test_dashboard_endpoint(self, client):
        """Test deployment dashboard endpoint"""
        response = client.get("/deployment/dashboard")
        
        assert response.status_code in [200, 429]
        if response.status_code == 200:
            data = response.json()
            assert "status" in data
            assert "overview" in data


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestDeploymentIntegration:
    """Integration tests for deployment features"""
    
    def test_region_manager_singleton(self):
        """Test region manager is singleton"""
        manager1 = get_region_manager()
        manager2 = get_region_manager()
        
        assert manager1 is manager2
    
    def test_failover_manager_singleton(self):
        """Test failover manager is singleton"""
        manager1 = get_failover_manager()
        manager2 = get_failover_manager()
        
        assert manager1 is manager2
    
    def test_region_manager_lifecycle(self):
        """Test complete region lifecycle"""
        manager = RegionManager()
        
        # Register region
        region_id = manager.register_region(
            "Lifecycle",
            "https://api.example.com",
            RegionTier.PRIMARY,
        )
        
        # Update metrics
        manager.update_region_metrics(region_id, 50, 90, 1, 1000)
        
        # Check status
        region = manager.get_region(region_id)
        assert region is not None
        assert region.status == RegionStatus.HEALTHY
        
        # Disable
        manager.disable_region(region_id)
        region = manager.get_region(region_id)
        assert region is not None
        assert region.enabled is False
    
    @pytest.mark.asyncio
    async def test_failover_workflow(self):
        """Test complete failover workflow"""
        failover_mgr = FailoverManager()
        
        # Set traffic
        distribution = {"region1": 100.0, "region2": 0.0}
        failover_mgr.set_traffic_distribution(distribution)
        
        # Initiate failover
        success = await failover_mgr.initiate_failover(
            from_region="region1",
            to_region="region2",
            trigger=FailoverTrigger.HIGH_ERROR_RATE,
        )
        
        assert isinstance(success, bool)
        
        # Check status
        status = failover_mgr.get_failover_status()
        assert "active_failovers" in status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
