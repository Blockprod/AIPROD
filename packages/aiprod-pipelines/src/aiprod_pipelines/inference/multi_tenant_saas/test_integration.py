"""
Comprehensive integration tests for Multi-Tenant SaaS platform.

Tests end-to-end workflows combining multiple components:
  - Complete user onboarding and tenant setup
  - Authentication and authorization flows
  - Billing and usage tracking integration
  - Job submission and execution
  - Feature rollout and configuration
  - Health monitoring and analytics
"""

import unittest
from datetime import datetime, timedelta

from aiprod_pipelines.inference.multi_tenant_saas.tenant_context import (
    TenantMetadata,
    TenantContext,
    TenantRegistry,
    TenantTier,
    TenantStatus,
)

from aiprod_pipelines.inference.multi_tenant_saas.authentication import (
    AuthenticationManager,
)

from aiprod_pipelines.inference.multi_tenant_saas.access_control import (
    RoleBasedAccessControl,
    ResourceType,
    Action,
)

from aiprod_pipelines.inference.multi_tenant_saas.api_gateway import (
    APIGateway,
    APIRequest,
    RateLimitConfig,
    RateLimitWindow,
)

from aiprod_pipelines.inference.multi_tenant_saas.usage_tracking import (
    UsageEventLogger,
    MeteringEngine,
    UsageEvent,
    UsageEventType,
)

from aiprod_pipelines.inference.multi_tenant_saas.billing import (
    BillingCalculator,
    PricingModel,
    SubscriptionPlan,
    BillingCycle,
)

from aiprod_pipelines.inference.multi_tenant_saas.job_manager import (
    JobScheduler,
    BatchJob,
    JobStatus,
)

from aiprod_pipelines.inference.multi_tenant_saas.configuration import (
    FeatureFlagManager,
    FeatureFlag,
    FeatureFlagType,
    ConfigurationManager,
)

from aiprod_pipelines.inference.multi_tenant_saas.monitoring import (
    MetricsCollector,
    AnomalyDetector,
    AnalyticsCollector,
    HealthMonitor,
)


class TestSaaSOnboarding(unittest.TestCase):
    """Test complete onboarding workflow."""
    
    def setUp(self):
        """Set up components."""
        self.tenant_registry = TenantRegistry()
        self.auth_manager = AuthenticationManager("secret_key")
        self.rbac = RoleBasedAccessControl()
    
    def test_customer_onboarding(self):
        """Test complete customer onboarding."""
        # 1. Create tenant
        tenant = TenantMetadata(
            tenant_id="acme_corp",
            organization_name="ACME Corporation",
            tier=TenantTier.PROFESSIONAL,
        )
        self.tenant_registry.register_tenant(tenant)
        
        verified_tenant = self.tenant_registry.get_tenant("acme_corp")
        self.assertIsNotNone(verified_tenant)
        
        # 2. Create admin API key
        key, api_key_obj = self.auth_manager.create_api_key(
            "acme_corp",
            "admin_key",
        )
        
        self.assertIsNotNone(key)
        self.assertTrue(api_key_obj.is_valid())
        
        # 3. Assign admin role
        self.rbac.assign_role_to_user("acme_corp", "admin@acme.com", "system_admin")
        
        roles = self.rbac.get_user_roles("acme_corp", "admin@acme.com")
        self.assertEqual(len(roles), 1)
        self.assertEqual(roles[0].role_id, "system_admin")
        
        # 4. Create regular user with user role
        self.rbac.assign_role_to_user("acme_corp", "user@acme.com", "system_user")
        
        # 5. Verify access control
        admin_can_manage = self.rbac.check_permission(
            "acme_corp",
            "admin@acme.com",
            ResourceType.USER_MANAGEMENT,
            Action.ADMIN,
        )
        self.assertTrue(admin_can_manage)
        
        user_cannot_manage = self.rbac.check_permission(
            "acme_corp",
            "user@acme.com",
            ResourceType.USER_MANAGEMENT,
            Action.ADMIN,
        )
        self.assertFalse(user_cannot_manage)


class TestBillingWorkflow(unittest.TestCase):
    """Test billing and metering workflow."""
    
    def setUp(self):
        """Set up billing components."""
        self.billing_calc = BillingCalculator()
        self.usage_logger = UsageEventLogger()
        self.metering = MeteringEngine()
        
        # Register pricing
        self.billing_calc.register_pricing_model(
            PricingModel("api_calls", base_price_per_unit=0.001)
        )
        self.billing_calc.register_pricing_model(
            PricingModel("video_generation", base_price_per_unit=0.50)
        )
        
        # Register plan
        self.billing_calc.register_subscription_plan(
            SubscriptionPlan(
                plan_id="professional",
                name="Professional",
                billing_cycle=BillingCycle.MONTHLY,
                price=99.99,
            )
        )
    
    def test_complete_billing_cycle(self):
        """Test complete monthly billing cycle."""
        tenant_id = "acme_corp"
        
        # 1. Log usage events
        for i in range(100):
            event = UsageEvent(
                event_id=f"evt_{i}",
                tenant_id=tenant_id,
                user_id="user_1",
                event_type=UsageEventType.API_CALL,
                resource_consumed=1.0,
                cost=0.001,
            )
            self.usage_logger.log_event(event)
        
        # Log video generation  
        for i in range(10):
            event = UsageEvent(
                event_id=f"video_{i}",
                tenant_id=tenant_id,
                user_id="user_1",
                event_type=UsageEventType.VIDEO_GENERATION,
                resource_consumed=1.0,  # 1 video
                cost=0.50,
            )
            self.usage_logger.log_event(event)
        
        # 2. Calculate charges
        api_charge, _ = self.billing_calc.calculate_usage_charge(
            "api_calls",
            100,  # 100 API calls
        )
        video_charge, _ = self.billing_calc.calculate_usage_charge(
            "video_generation",
            10,  # 10 videos
        )
        
        self.assertEqual(api_charge, 0.10)  # 100 * 0.001
        self.assertEqual(video_charge, 5.0)  # 10 * 0.50
        
        # 3. Generate invoice
        invoice = self.billing_calc.generate_invoice(
            tenant_id=tenant_id,
            period_start=datetime.utcnow() - timedelta(days=30),
            period_end=datetime.utcnow(),
            usage_charges={"api_calls": api_charge, "video_generation": video_charge},
            plan_charge=99.99,
            tax_rate=0.1,
        )
        
        # 4. Verify invoice
        self.assertIsNotNone(invoice)
        self.assertEqual(invoice.subtotal, 99.99 + api_charge + video_charge)
    
    def test_quota_enforcement(self):
        """Test usage quota enforcement."""
        tenant_id = "acme_corp"
        
        # Record usage
        self.metering.record_usage(tenant_id, "videos_per_month", 50)
        
        # Check against quota
        within_quota, current = self.metering.check_quota(
            tenant_id,
            "videos_per_month",
            100,  # 100 video quota
        )
        
        self.assertTrue(within_quota)
        
        # Exceed quota
        self.metering.record_usage(tenant_id, "videos_per_month", 75)
        
        within_quota, current = self.metering.check_quota(
            tenant_id,
            "videos_per_month",
            100,
        )
        
        self.assertFalse(within_quota)


class TestJobWorkflow(unittest.TestCase):
    """Test job submission and execution workflow."""
    
    def setUp(self):
        """Set up job scheduler."""
        self.scheduler = JobScheduler(max_concurrent_jobs=5)
    
    def test_video_generation_job_workflow(self):
        """Test video generation job workflow."""
        # 1. Submit job
        job = BatchJob(
            job_id="vid_gen_1",
            tenant_id="acme_corp",
            user_id="user_1",
            job_type="video_generation",
            parameters={
                "prompt": "A cat walking through a forest",
                "num_frames": 120,
                "fps": 24,
            },
            estimated_duration_seconds=60.0,
        )
        
        job_id = self.scheduler.submit_job(job)
        self.assertEqual(job_id, "vid_gen_1")
        
        # 2. Check job status
        status = self.scheduler.get_job_status("vid_gen_1")
        self.assertEqual(status, JobStatus.QUEUED)
        
        # 3. Update progress
        progress = self.scheduler._progress_tracker
        progress.update_progress("vid_gen_1", 25.0, "Starting denoising...")
        
        # 4. Retrieve job with progress
        current_job = self.scheduler.get_job("vid_gen_1")
        self.assertIsNotNone(current_job)
    
    def test_batch_job_submission(self):
        """Test batch job submission."""
        jobs = []
        for i in range(5):
            job = BatchJob(
                job_id=f"batch_{i}",
                tenant_id="acme_corp",
                user_id="user_1",
                job_type="video_generation",
            )
            self.scheduler.submit_job(job)
            jobs.append(job)
        
        # Verify all submitted
        tenant_jobs = self.scheduler.get_tenant_jobs("acme_corp")
        self.assertEqual(len(tenant_jobs), 5)


class TestFeatureRollout(unittest.TestCase):
    """Test feature flag and rollout workflow."""
    
    def setUp(self):
        """Set up feature flag manager."""
        self.flag_manager = FeatureFlagManager()
    
    def test_gradual_feature_rollout(self):
        """Test gradual feature rollout."""
        # 1. Create feature flag for new capability
        flag = FeatureFlag(
            flag_id="lora_training",
            name="LoRA Fine-tuning",
            description="User-customizable LoRA training",
            enabled=True,
            flag_type=FeatureFlagType.PERCENTAGE,
            rollout_percentage=10.0,
        )
        
        self.flag_manager.create_flag(flag)
        
        # 2. Initially 10% rollout
        enabled_count_10pct = sum(
            1 for i in range(100)
            if self.flag_manager.is_feature_enabled("lora_training", f"tenant_{i}")
        )
        
        self.assertGreater(enabled_count_10pct, 0)
        self.assertLess(enabled_count_10pct, 30)  # Should be around 10%
        
        # 3. Increase to 50% rollout
        self.flag_manager.set_rollout_percentage("lora_training", 50.0)
        
        enabled_count_50pct = sum(
            1 for i in range(100)
            if self.flag_manager.is_feature_enabled("lora_training", f"tenant_{i}")
        )
        
        self.assertGreater(enabled_count_50pct, enabled_count_10pct)
    
    def test_targeted_feature_rollout(self):
        """Test targeted rollout for specific tenants."""
        flag = FeatureFlag(
            flag_id="beta_feature",
            name="Beta Feature",
            enabled=True,
            flag_type=FeatureFlagType.USER_LIST,
        )
        
        self.flag_manager.create_flag(flag)
        
        # Add specific tenants
        self.flag_manager.add_enabled_tenant("beta_feature", "acme_corp")
        self.flag_manager.add_enabled_tenant("beta_feature", "big_tech_inc")
        
        # Check enabled
        self.assertTrue(
            self.flag_manager.is_feature_enabled("beta_feature", "acme_corp")
        )
        self.assertTrue(
            self.flag_manager.is_feature_enabled("beta_feature", "big_tech_inc")
        )
        
        # Check not enabled for others
        self.assertFalse(
            self.flag_manager.is_feature_enabled("beta_feature", "other_company")
        )


class TestPlatformMonitoring(unittest.TestCase):
    """Test platform monitoring and analytics."""
    
    def setUp(self):
        """Set up monitoring components."""
        self.metrics_collector = MetricsCollector()
        self.anomaly_detector = AnomalyDetector()
        self.analytics = AnalyticsCollector()
        self.health_monitor = HealthMonitor(self.metrics_collector, self.anomaly_detector)
    
    def test_metrics_collection(self):
        """Test metric collection and analysis."""
        # Collect metrics
        for i in range(100):
            self.metrics_collector.record_metric(
                "response_time_ms",
                value=50.0 + i * 0.5,  # Gradually increasing
            )
        
        # Get statistics
        stats = self.metrics_collector.get_metric_statistics("response_time_ms")
        
        self.assertIsNotNone(stats)
        self.assertEqual(stats["count"], 100)
        self.assertGreater(stats["mean"], 0)
        self.assertGreater(stats["max"], stats["min"])
    
    def test_anomaly_detection(self):
        """Test anomaly detection."""
        # Establish baseline
        baseline_values = [100.0] * 50
        self.anomaly_detector.establish_baseline("error_rate", baseline_values)
        
        # Normal value
        is_anomaly, z_score = self.anomaly_detector.detect_anomaly("error_rate", 105.0)
        self.assertFalse(is_anomaly)
        
        # Anomalous value (far from baseline)
        is_anomaly, z_score = self.anomaly_detector.detect_anomaly("error_rate", 500.0)
        self.assertTrue(is_anomaly)
    
    def test_event_analytics(self):
        """Test event tracking and analytics."""
        # Track events
        for i in range(50):
            self.analytics.track_event(
                "video_generation_completed",
                tenant_id="t1",
                properties={"duration_seconds": 30.0, "status": "success"},
            )
        
        for i in range(10):
            self.analytics.track_event(
                "video_generation_failed",
                tenant_id="t1",
                properties={"error": "Timeout"},
            )
        
        # Get tenant analytics
        analytics = self.analytics.get_tenant_analytics("t1", hours=24)
        
        self.assertEqual(analytics["total_events"], 60)
        self.assertEqual(
            analytics["event_summary"]["video_generation_completed"],
            50,
        )
        self.assertEqual(
            analytics["event_summary"]["video_generation_failed"],
            10,
        )
    
    def test_platform_health(self):
        """Test platform health monitoring."""
        # Record some metrics
        for i in range(50):
            error_rate = 2.0 if i < 40 else 15.0  # Degradation near end
            self.metrics_collector.record_metric("error_rate", error_rate)
        
        health = self.health_monitor.get_platform_health()
        
        self.assertIsNotNone(health["status"])
        self.assertIn("issues", health)


class TestMultiTenantIsolation(unittest.TestCase):
    """Test multi-tenant isolation."""
    
    def test_data_isolation(self):
        """Test tenant data isolation."""
        registry = TenantRegistry()
        
        # Create two tenants
        t1 = TenantMetadata(
            tenant_id="t1",
            organization_name="Org 1",
            tier=TenantTier.FREE,
        )
        t2 = TenantMetadata(
            tenant_id="t2",
            organization_name="Org 2",
            tier=TenantTier.PROFESSIONAL,
        )
        
        registry.register_tenant(t1)
        registry.register_tenant(t2)
        
        # Add features to t1
        registry.add_feature("t1", "lora")
        
        # Verify t2 doesn't have feature
        t1_retrieved = registry.get_tenant("t1")
        t2_retrieved = registry.get_tenant("t2")
        
        self.assertTrue(t1_retrieved.has_feature("lora"))
        self.assertFalse(t2_retrieved.has_feature("lora"))
    
    def test_rbac_isolation(self):
        """Test RBAC isolation between tenants."""
        rbac = RoleBasedAccessControl()
        
        # Assign different roles
        rbac.assign_role_to_user("t1", "user1", "system_admin")
        rbac.assign_role_to_user("t2", "user1", "system_user")
        
        # Check permissions in each tenant
        t1_perm = rbac.check_permission(
            "t1",
            "user1",
            ResourceType.BILLING,
            Action.ADMIN,
        )
        
        t2_perm = rbac.check_permission(
            "t2",
            "user1",
            ResourceType.BILLING,
            Action.ADMIN,
        )
        
        self.assertTrue(t1_perm)
        self.assertFalse(t2_perm)


if __name__ == "__main__":
    unittest.main()
