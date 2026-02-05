"""
Tests for Disaster Recovery (DR) management
Tests failover scenarios, RTO/RPO, and recovery procedures
"""

import pytest
from datetime import datetime, timedelta
from src.infra.dr_manager import (
    DRManager, DisasterStatus, FailoverType, RecoveryMetrics,
    DRScenario, get_dr_manager
)


class TestDRScenarios:
    """Test DR scenario definitions"""
    
    def test_all_scenarios_defined(self):
        """Test that all scenarios are defined"""
        manager = DRManager()
        scenarios = manager.get_scenarios()
        assert len(scenarios) == 6
    
    def test_scenario_database_failure(self):
        """Test database failure scenario exists"""
        manager = DRManager()
        scenario = manager.get_scenario(FailoverType.DATABASE_FAILURE)
        assert scenario is not None
        assert scenario.name == "Database Failover"
        assert len(scenario.steps) > 0
        assert scenario.rto_target == 120
    
    def test_scenario_api_failure(self):
        """Test API failure scenario"""
        manager = DRManager()
        scenario = manager.get_scenario(FailoverType.API_SERVICE_FAILURE)
        assert scenario is not None
        assert scenario.rto_target == 60
    
    def test_scenario_region_failure(self):
        """Test regional failure scenario"""
        manager = DRManager()
        scenario = manager.get_scenario(FailoverType.REGION_FAILURE)
        assert scenario is not None
        assert scenario.rpo_target == 60
    
    def test_scenario_complete_outage(self):
        """Test complete outage scenario"""
        manager = DRManager()
        scenario = manager.get_scenario(FailoverType.COMPLETE_OUTAGE)
        assert scenario is not None
        assert scenario.rto_target == 900  # 15 minutes
    
    def test_scenario_data_corruption(self):
        """Test data corruption scenario"""
        manager = DRManager()
        scenario = manager.get_scenario(FailoverType.DATA_CORRUPTION)
        assert scenario is not None
        assert scenario.name == "Data Corruption Recovery"
    
    def test_scenario_security_breach(self):
        """Test security breach scenario"""
        manager = DRManager()
        scenario = manager.get_scenario(FailoverType.SECURITY_BREACH)
        assert scenario is not None
        assert scenario.priority == "high"


class TestRecoveryMetrics:
    """Test RecoveryMetrics data class"""
    
    def test_recovery_metrics_creation(self):
        """Test creating recovery metrics"""
        start = datetime.utcnow()
        metrics = RecoveryMetrics(
            scenario=FailoverType.API_SERVICE_FAILURE,
            start_time=start
        )
        assert metrics.scenario == FailoverType.API_SERVICE_FAILURE
        assert metrics.duration is None  # Not yet completed
    
    def test_recovery_duration(self):
        """Test calculating recovery duration"""
        start = datetime.utcnow()
        end = start + timedelta(seconds=45)
        
        metrics = RecoveryMetrics(
            scenario=FailoverType.API_SERVICE_FAILURE,
            start_time=start,
            end_time=end
        )
        assert metrics.duration == 45
    
    def test_rto_met_success(self):
        """Test RTO target success"""
        start = datetime.utcnow()
        end = start + timedelta(seconds=50)  # Under 60s target
        
        metrics = RecoveryMetrics(
            scenario=FailoverType.API_SERVICE_FAILURE,
            start_time=start,
            end_time=end,
            rto_target=60
        )
        assert metrics.rto_met is True
    
    def test_rto_met_failure(self):
        """Test RTO target failure"""
        start = datetime.utcnow()
        end = start + timedelta(seconds=90)  # Over 60s target
        
        metrics = RecoveryMetrics(
            scenario=FailoverType.API_SERVICE_FAILURE,
            start_time=start,
            end_time=end,
            rto_target=60
        )
        assert metrics.rto_met is False
    
    def test_rpo_calculation(self):
        """Test RPO calculation"""
        start = datetime.utcnow()
        end = start + timedelta(seconds=30)
        
        metrics = RecoveryMetrics(
            scenario=FailoverType.DATABASE_FAILURE,
            start_time=start,
            end_time=end,
            rpo_target=60,
            data_loss=1000,  # 1KB
            services_affected=["api", "worker"]
        )
        assert metrics.rpo_actual > 0
    
    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary"""
        start = datetime.utcnow()
        end = start + timedelta(seconds=120)
        
        metrics = RecoveryMetrics(
            scenario=FailoverType.DATABASE_FAILURE,
            start_time=start,
            end_time=end,
            rto_target=300,
            rpo_target=60,
        )
        
        data = metrics.to_dict()
        assert data["scenario"] == "database_failure"
        assert data["duration_seconds"] == 120
        assert "rto_actual" in data
        assert "rpo_actual" in data


class TestDRManager:
    """Test DRManager functionality"""
    
    def test_dr_manager_creation(self):
        """Test creating DR manager"""
        manager = DRManager()
        assert manager is not None
        assert manager.current_status == DisasterStatus.NORMAL
    
    def test_get_scenario(self):
        """Test getting specific scenario"""
        manager = DRManager()
        scenario = manager.get_scenario(FailoverType.DATABASE_FAILURE)
        assert scenario is not None
        assert isinstance(scenario, DRScenario)
    
    def test_start_recovery(self):
        """Test starting recovery"""
        manager = DRManager()
        metrics = manager.start_recovery(FailoverType.API_SERVICE_FAILURE)
        
        assert metrics.scenario == FailoverType.API_SERVICE_FAILURE
        assert manager.current_status == DisasterStatus.RECOVERING
    
    def test_complete_recovery(self):
        """Test completing recovery"""
        manager = DRManager()
        metrics = manager.start_recovery(FailoverType.DATABASE_FAILURE)
        
        # Simulate recovery
        import time
        time.sleep(0.1)
        
        completed = manager.complete_recovery(metrics)
        assert completed.end_time is not None
        assert manager.current_status == DisasterStatus.RECOVERED
    
    def test_get_recovery_runbook(self):
        """Test getting recovery runbook"""
        manager = DRManager()
        runbook = manager.get_recovery_runbook(FailoverType.API_SERVICE_FAILURE)
        
        assert "steps" in runbook
        assert "expected_behavior" in runbook
        assert "rto_target" in runbook
        assert len(runbook["steps"]) > 0
    
    def test_runbook_database_failure(self):
        """Test database failure runbook"""
        manager = DRManager()
        runbook = manager.get_recovery_runbook(FailoverType.DATABASE_FAILURE)
        
        assert "Failover" in runbook["title"]
        assert any("replica" in step.lower() for step in runbook["steps"])
    
    def test_get_scenarios(self):
        """Test getting all scenarios"""
        manager = DRManager()
        scenarios = manager.get_scenarios()
        
        assert len(scenarios) == 6
        assert all(isinstance(s, DRScenario) for s in scenarios)
    
    def test_metrics_summary_empty(self):
        """Test metrics summary with no drills"""
        manager = DRManager()
        summary = manager.get_metrics_summary()
        
        assert summary["total_drills"] == 0
        assert summary["successful_recoveries"] == 0
    
    def test_metrics_summary_with_drills(self):
        """Test metrics summary with completed drills"""
        manager = DRManager()
        
        # Complete multiple drills
        for i in range(3):
            start = datetime.utcnow()
            end = start + timedelta(seconds=50)  # Within RTO target
            
            metrics = RecoveryMetrics(
                scenario=FailoverType.API_SERVICE_FAILURE,
                start_time=start,
                end_time=end,
                rto_target=60,
                rpo_target=10,
                data_loss=100,
                requests_failed=0
            )
            manager.metrics.append(metrics)
        
        summary = manager.get_metrics_summary()
        assert summary["total_drills"] == 3
        assert summary["successful_recoveries"] == 3
        assert summary["success_rate"] == 1.0
    
    def test_metrics_summary_mixed_results(self):
        """Test metrics summary with mixed pass/fail"""
        manager = DRManager()
        
        # Successful recovery
        start1 = datetime.utcnow()
        end1 = start1 + timedelta(seconds=50)
        metrics1 = RecoveryMetrics(
            scenario=FailoverType.API_SERVICE_FAILURE,
            start_time=start1,
            end_time=end1,
            rto_target=60
        )
        manager.metrics.append(metrics1)
        
        # Failed recovery (over RTO)
        start2 = datetime.utcnow()
        end2 = start2 + timedelta(seconds=120)
        metrics2 = RecoveryMetrics(
            scenario=FailoverType.DATABASE_FAILURE,
            start_time=start2,
            end_time=end2,
            rto_target=60
        )
        manager.metrics.append(metrics2)
        
        summary = manager.get_metrics_summary()
        assert summary["total_drills"] == 2
        assert summary["successful_recoveries"] == 1
        assert summary["failed_recoveries"] == 1
        assert summary["success_rate"] == 0.5
    
    def test_create_all_runbooks(self):
        """Test creating all runbooks"""
        manager = DRManager()
        runbooks = manager.create_runbooks()
        
        assert len(runbooks) == 6
        assert "database_failure" in runbooks
        assert "api_service_failure" in runbooks
        assert "region_failure" in runbooks
        assert "complete_outage" in runbooks
        assert "data_corruption" in runbooks
        assert "security_breach" in runbooks


class TestDRIntegration:
    """Integration tests for DR system"""
    
    def test_complete_recovery_workflow(self):
        """Test complete recovery workflow"""
        manager = DRManager()
        
        # Start recovery
        metrics = manager.start_recovery(FailoverType.DATABASE_FAILURE)
        assert manager.current_status == DisasterStatus.RECOVERING
        
        # Simulate recovery - longer sleep to ensure measurable duration
        import time
        time.sleep(0.2)
        
        # Complete recovery
        metrics = manager.complete_recovery(metrics)
        assert metrics.end_time is not None
        assert manager.current_status == DisasterStatus.RECOVERED
        assert metrics.duration is not None  # Should have a measurable duration
    
    def test_scenario_recovery_targets(self):
        """Test all scenarios have recovery targets"""
        manager = DRManager()
        
        for scenario in manager.get_scenarios():
            assert scenario.rto_target > 0
            assert scenario.rpo_target >= 0
            assert scenario.rto_target > scenario.rpo_target or scenario.rpo_target == 60
    
    def test_dr_manager_singleton(self):
        """Test DR manager singleton"""
        manager1 = get_dr_manager()
        manager2 = get_dr_manager()
        assert manager1 is manager2
    
    def test_high_priority_scenarios(self):
        """Test that critical scenarios are high priority"""
        manager = DRManager()
        
        critical_scenarios = [
            FailoverType.COMPLETE_OUTAGE,
            FailoverType.REGION_FAILURE,
            FailoverType.SECURITY_BREACH,
        ]
        
        for scenario_type in critical_scenarios:
            scenario = manager.get_scenario(scenario_type)
            assert scenario is not None
            assert scenario.priority in ["high", "critical"]
    
    def test_recovery_time_progression(self):
        """Test realistic recovery time progression"""
        manager = DRManager()
        
        # Faster recovery for API failure
        api_scenario = manager.get_scenario(FailoverType.API_SERVICE_FAILURE)
        db_scenario = manager.get_scenario(FailoverType.DATABASE_FAILURE)
        region_scenario = manager.get_scenario(FailoverType.REGION_FAILURE)
        
        # Verify scenarios are not None
        assert api_scenario is not None
        assert db_scenario is not None
        assert region_scenario is not None
        
        # API should be fastest
        assert api_scenario.rto_target < db_scenario.rto_target
        # DB should be faster than region
        assert db_scenario.rto_target < region_scenario.rto_target
    
    def test_runbook_completeness(self):
        """Test runbooks have all required sections"""
        manager = DRManager()
        
        for scenario_type in FailoverType:
            runbook = manager.get_recovery_runbook(scenario_type)
            
            assert "title" in runbook
            assert "description" in runbook
            assert "steps" in runbook
            assert "expected_behavior" in runbook
            assert "recovery_steps" in runbook
            assert len(runbook["steps"]) >= 3
            assert len(runbook["recovery_steps"]) >= 2


class TestDRMetricsComparison:
    """Test comparison and analysis of DR metrics"""
    
    def test_compare_scenario_performance(self):
        """Test comparing performance across scenarios"""
        manager = DRManager()
        
        # Run recovery for different scenarios
        scenarios = [
            (FailoverType.API_SERVICE_FAILURE, 45),
            (FailoverType.DATABASE_FAILURE, 90),
            (FailoverType.REGION_FAILURE, 200),
        ]
        
        for scenario_type, duration in scenarios:
            start = datetime.utcnow()
            end = start + timedelta(seconds=duration)
            
            metrics = RecoveryMetrics(
                scenario=scenario_type,
                start_time=start,
                end_time=end
            )
            manager.metrics.append(metrics)
        
        summary = manager.get_metrics_summary()
        assert summary["total_drills"] == 3
        
        # Verify average RTO is reasonable
        avg_rto = summary["average_rto_seconds"]
        assert avg_rto > 0
        assert avg_rto < 500
