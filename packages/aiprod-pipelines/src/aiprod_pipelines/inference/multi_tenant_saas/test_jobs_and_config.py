"""
Comprehensive tests for job management and configuration.

Tests:
  - Batch jobs and status
  - Job scheduler and queuing
  - Job progress tracking
  - Feature flags
  - Tenant configuration
  - Configuration management
"""

import unittest
from datetime import datetime, timedelta

from aiprod_pipelines.inference.multi_tenant_saas.job_manager import (
    BatchJob,
    JobStatus,
    JobPriority,
    JobProgressTracker,
    JobScheduler,
    JobManagementPortal,
)

from aiprod_pipelines.inference.multi_tenant_saas.configuration import (
    FeatureFlag,
    FeatureFlagType,
    RolloutStatus,
    FeatureFlagManager,
    TenantConfig,
    ConfigurationManager,
)


class TestBatchJob(unittest.TestCase):
    """Test batch job."""
    
    def test_create_job(self):
        """Test job creation."""
        job = BatchJob(
            job_id="job_1",
            tenant_id="t1",
            user_id="user_1",
            job_type="video_generation",
        )
        
        self.assertEqual(job.job_id, "job_1")
        self.assertEqual(job.status, JobStatus.QUEUED)
        self.assertFalse(job.is_running())
        self.assertFalse(job.is_completed())
    
    def test_job_duration(self):
        """Test job duration calculation."""
        job = BatchJob(
            job_id="job_1",
            tenant_id="t1",
            user_id="user_1",
            job_type="video_generation",
        )
        
        now = datetime.utcnow()
        job.started_at = now - timedelta(seconds=30)
        job.completed_at = now
        
        duration = job.get_duration_seconds()
        self.assertAlmostEqual(duration, 30.0, delta=1.0)
    
    def test_job_retry(self):
        """Test job retry capability."""
        job = BatchJob(
            job_id="job_1",
            tenant_id="t1",
            user_id="user_1",
            job_type="video_generation",
            max_retries=3,
        )
        
        job.status = JobStatus.FAILED
        self.assertTrue(job.can_retry())
        
        job.retry_count = 3
        self.assertFalse(job.can_retry())
    
    def test_job_to_dict(self):
        """Test job serialization."""
        job = BatchJob(
            job_id="job_1",
            tenant_id="t1",
            user_id="user_1",
            job_type="video_generation",
            progress_percentage=50.0,
        )
        
        job_dict = job.to_dict()
        self.assertEqual(job_dict["job_id"], "job_1")
        self.assertEqual(job_dict["progress_percentage"], 50.0)
        self.assertEqual(job_dict["status"], "queued")


class TestJobProgressTracker(unittest.TestCase):
    """Test job progress tracking."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = JobProgressTracker()
    
    def test_update_progress(self):
        """Test updating job progress."""
        self.tracker.update_progress("job_1", 50.0, "Processing...")
        
        progress = self.tracker.get_progress("job_1")
        self.assertIsNotNone(progress)
        self.assertEqual(progress["percentage"], 50.0)
        self.assertEqual(progress["message"], "Processing...")
    
    def test_progress_clamping(self):
        """Test progress percentage clamping."""
        self.tracker.update_progress("job_1", 150.0)
        progress = self.tracker.get_progress("job_1")
        self.assertEqual(progress["percentage"], 100.0)
        
        self.tracker.update_progress("job_2", -50.0)
        progress = self.tracker.get_progress("job_2")
        self.assertEqual(progress["percentage"], 0.0)
    
    def test_progress_subscription(self):
        """Test progress subscription."""
        updates = []
        
        def callback(progress):
            updates.append(progress)
        
        self.tracker.subscribe_to_progress("job_1", callback)
        
        self.tracker.update_progress("job_1", 25.0)
        self.tracker.update_progress("job_1", 50.0)
        
        self.assertEqual(len(updates), 2)
        self.assertEqual(updates[0]["percentage"], 25.0)
        self.assertEqual(updates[1]["percentage"], 50.0)


class TestJobScheduler(unittest.TestCase):
    """Test job scheduler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scheduler = JobScheduler(max_concurrent_jobs=5)
    
    def test_submit_job(self):
        """Test submitting job."""
        job = BatchJob(
            job_id="job_1",
            tenant_id="t1",
            user_id="user_1",
            job_type="video_generation",
        )
        
        job_id = self.scheduler.submit_job(job)
        self.assertEqual(job_id, "job_1")
    
    def test_get_job_status(self):
        """Test getting job status."""
        job = BatchJob(
            job_id="job_1",
            tenant_id="t1",
            user_id="user_1",
            job_type="video_generation",
        )
        
        self.scheduler.submit_job(job)
        status = self.scheduler.get_job_status("job_1")
        
        self.assertEqual(status, JobStatus.QUEUED)
    
    def test_cancel_job(self):
        """Test cancelling job."""
        job = BatchJob(
            job_id="job_1",
            tenant_id="t1",
            user_id="user_1",
            job_type="video_generation",
        )
        
        self.scheduler.submit_job(job)
        cancelled = self.scheduler.cancel_job("job_1")
        
        self.assertTrue(cancelled)
        status = self.scheduler.get_job_status("job_1")
        self.assertEqual(status, JobStatus.CANCELLED)
    
    def test_get_tenant_jobs(self):
        """Test getting tenant jobs."""
        for i in range(3):
            job = BatchJob(
                job_id=f"job_{i}",
                tenant_id="t1",
                user_id="user_1",
                job_type="video_generation",
            )
            self.scheduler.submit_job(job)
        
        jobs = self.scheduler.get_tenant_jobs("t1")
        self.assertEqual(len(jobs), 3)
    
    def test_job_priority_queuing(self):
        """Test job priority in queue."""
        low_job = BatchJob(
            job_id="low",
            tenant_id="t1",
            user_id="user_1",
            job_type="video_generation",
            priority=JobPriority.LOW,
        )
        
        high_job = BatchJob(
            job_id="high",
            tenant_id="t1",
            user_id="user_1",
            job_type="video_generation",
            priority=JobPriority.HIGH,
        )
        
        # Submit low priority first
        self.scheduler.submit_job(low_job)
        self.scheduler.submit_job(high_job)
        
        # Queue should be sorted by priority
        self.assertEqual(len(self.scheduler._job_queue), 2)
    
    def test_scheduler_stats(self):
        """Test scheduler statistics."""
        job = BatchJob(
            job_id="job_1",
            tenant_id="t1",
            user_id="user_1",
            job_type="video_generation",
        )
        
        self.scheduler.submit_job(job)
        stats = self.scheduler.get_scheduler_stats()
        
        self.assertEqual(stats["queued_jobs"], 1)
        self.assertEqual(stats["running_jobs"], 0)
        self.assertEqual(stats["max_concurrent"], 5)


class TestFeatureFlag(unittest.TestCase):
    """Test feature flags."""
    
    def test_create_flag(self):
        """Test flag creation."""
        flag = FeatureFlag(
            flag_id="feature_lora",
            name="LoRA Training",
            enabled=True,
        )
        
        self.assertEqual(flag.flag_id, "feature_lora")
        self.assertTrue(flag.enabled)
    
    def test_boolean_flag(self):
        """Test boolean flag."""
        flag = FeatureFlag(
            flag_id="f1",
            name="Feature1",
            flag_type=FeatureFlagType.BOOLEAN,
            enabled=True,
        )
        
        self.assertTrue(flag.is_enabled_for_tenant("any_tenant"))
    
    def test_percentage_rollout(self):
        """Test percentage-based rollout."""
        flag = FeatureFlag(
            flag_id="f1",
            name="Feature1",
            flag_type=FeatureFlagType.PERCENTAGE,
            enabled=True,
            rollout_percentage=50.0,
        )
        
        # Should enable for approximately half of tenants
        enabled_count = sum(
            1 for i in range(100)
            if flag.is_enabled_for_tenant(f"tenant_{i}")
        )
        
        # Should be around 50, allow ±20% variance
        self.assertGreater(enabled_count, 30)
        self.assertLess(enabled_count, 70)
    
    def test_user_list_rollout(self):
        """Test user list targeted rollout."""
        flag = FeatureFlag(
            flag_id="f1",
            name="Feature1",
            flag_type=FeatureFlagType.USER_LIST,
            enabled=True,
        )
        
        flag.enabled_tenants.add("t1")
        flag.enabled_tenants.add("t2")
        
        self.assertTrue(flag.is_enabled_for_tenant("t1"))
        self.assertTrue(flag.is_enabled_for_tenant("t2"))
        self.assertFalse(flag.is_enabled_for_tenant("t3"))


class TestFeatureFlagManager(unittest.TestCase):
    """Test feature flag manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = FeatureFlagManager()
    
    def test_create_flag(self):
        """Test creating flag."""
        flag = FeatureFlag(
            flag_id="f1",
            name="Feature1",
        )
        
        self.manager.create_flag(flag)
        retrieved = self.manager.get_flag("f1")
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.flag_id, "f1")
    
    def test_enable_disable_flag(self):
        """Test enabling and disabling flags."""
        flag = FeatureFlag(flag_id="f1", name="Feature1", enabled=False)
        self.manager.create_flag(flag)
        
        self.manager.enable_flag("f1")
        flag = self.manager.get_flag("f1")
        self.assertTrue(flag.enabled)
        
        self.manager.disable_flag("f1")
        flag = self.manager.get_flag("f1")
        self.assertFalse(flag.enabled)
    
    def test_set_rollout_percentage(self):
        """Test setting rollout percentage."""
        flag = FeatureFlag(flag_id="f1", name="Feature1", enabled=True)
        self.manager.create_flag(flag)
        
        self.manager.set_rollout_percentage("f1", 25.0)
        flag = self.manager.get_flag("f1")
        self.assertEqual(flag.rollout_percentage, 25.0)


class TestTenantConfig(unittest.TestCase):
    """Test tenant configuration."""
    
    def test_create_config(self):
        """Test creating tenant config."""
        config = TenantConfig(tenant_id="t1")
        
        self.assertEqual(config.tenant_id, "t1")
        self.assertEqual(config.default_model, "ltx-video-2")
    
    def test_feature_settings(self):
        """Test feature settings."""
        config = TenantConfig(tenant_id="t1")
        
        config.set_feature_enabled("lora", True)
        self.assertTrue(config.get_feature_enabled("lora"))
        
        config.set_feature_enabled("lora", False)
        self.assertFalse(config.get_feature_enabled("lora"))
    
    def test_custom_settings(self):
        """Test custom settings."""
        config = TenantConfig(tenant_id="t1")
        
        config.set_custom_setting("webhook_timeout", 30)
        self.assertEqual(config.get_custom_setting("webhook_timeout"), 30)


class TestConfigurationManager(unittest.TestCase):
    """Test configuration manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manager = ConfigurationManager()
    
    def test_create_config(self):
        """Test creating tenant config."""
        config = self.manager.create_config("t1")
        
        self.assertEqual(config.tenant_id, "t1")
        self.assertIsNotNone(config)
    
    def test_update_config(self):
        """Test updating config."""
        config = self.manager.create_config("t1")
        config.set_feature_enabled("lora", True)
        
        self.manager.update_config(config)
        
        retrieved = self.manager.get_config("t1")
        self.assertTrue(retrieved.get_feature_enabled("lora"))


if __name__ == "__main__":
    unittest.main()
