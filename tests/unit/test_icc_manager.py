"""
Tests for ICC Manager - Interactive Creative Control endpoints.
"""

import pytest
import asyncio
from datetime import datetime
from src.api.icc_manager import JobManager, JobState, Job


class TestJobManager:
    """Tests for JobManager class."""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh JobManager for each test."""
        return JobManager()
    
    @pytest.mark.asyncio
    async def test_create_job_minimal(self, manager):
        """Test creating a job with minimal data."""
        job = await manager.create_job(content="Test video brief")
        
        assert job.id is not None
        assert job.content == "Test video brief"
        assert job.state == JobState.CREATED
        assert job.preset is None
        
    @pytest.mark.asyncio
    async def test_create_job_with_preset(self, manager):
        """Test creating a job with preset."""
        job = await manager.create_job(
            content="Brand video",
            preset="brand_campaign"
        )
        
        assert job.preset == "brand_campaign"
        assert job.state == JobState.CREATED
        
    @pytest.mark.asyncio
    async def test_create_job_with_all_options(self, manager):
        """Test creating a job with all options."""
        job = await manager.create_job(
            content="Premium video",
            preset="premium_spot",
            priority="high",
            lang="fr",
            brand_id="brand-123"
        )
        
        assert job.content == "Premium video"
        assert job.preset == "premium_spot"
        assert job.priority == "high"
        assert job.lang == "fr"
        assert job.brand_id == "brand-123"
        
    @pytest.mark.asyncio
    async def test_create_job_generates_unique_ids(self, manager):
        """Test that each job gets a unique ID."""
        job1 = await manager.create_job(content="Brief 1")
        job2 = await manager.create_job(content="Brief 2")
        
        assert job1.id != job2.id
        
    @pytest.mark.asyncio
    async def test_get_job_existing(self, manager):
        """Test getting an existing job."""
        created = await manager.create_job(content="Test")
        job_id = created.id
        
        retrieved = await manager.get_job(job_id)
        
        assert retrieved is not None
        assert retrieved.id == job_id
        assert retrieved.content == "Test"
        
    @pytest.mark.asyncio
    async def test_get_job_nonexistent(self, manager):
        """Test getting a non-existent job returns None."""
        result = await manager.get_job("nonexistent-id")
        assert result is None
        
    @pytest.mark.asyncio
    async def test_update_job_state(self, manager):
        """Test updating job state."""
        job = await manager.create_job(content="Test")
        job_id = job.id
        
        updated = await manager.update_job_state(job_id, JobState.ANALYZING)
        
        assert updated is not None
        assert updated.state == JobState.ANALYZING
        
    @pytest.mark.asyncio
    async def test_update_job_state_nonexistent(self, manager):
        """Test updating state of non-existent job."""
        result = await manager.update_job_state("fake-id", JobState.RENDERING)
        assert result is None
        
    @pytest.mark.asyncio
    async def test_set_manifest(self, manager):
        """Test setting job manifest."""
        job = await manager.create_job(content="Test")
        job_id = job.id
        
        manifest = {"title": "Test Video", "shots": []}
        updated = await manager.set_manifest(job_id, manifest)
        
        assert updated is not None
        assert updated.production_manifest == manifest
        
    @pytest.mark.asyncio
    async def test_set_manifest_with_consistency_markers(self, manager):
        """Test setting manifest with consistency markers."""
        job = await manager.create_job(content="Test")
        
        manifest = {"title": "Test"}
        markers = {"style_hash": "abc123"}
        updated = await manager.set_manifest(job.id, manifest, markers)
        
        assert updated.production_manifest == manifest
        assert updated.consistency_markers == markers
        
    @pytest.mark.asyncio
    async def test_update_manifest_requires_waiting_approval_state(self, manager):
        """Test that manifest updates require WAITING_APPROVAL state."""
        job = await manager.create_job(content="Test")
        
        # Job is in CREATED state, not WAITING_APPROVAL
        result = await manager.update_manifest(job.id, {"shot_list": []})
        
        # Should return None because state is wrong
        assert result is None
        
    @pytest.mark.asyncio
    async def test_update_manifest_in_waiting_approval(self, manager):
        """Test updating manifest in WAITING_APPROVAL state."""
        job = await manager.create_job(content="Test")
        
        # Set initial manifest and move to WAITING_APPROVAL
        await manager.set_manifest(job.id, {"shot_list": ["shot1"]})
        await manager.update_job_state(job.id, JobState.WAITING_APPROVAL)
        
        # Now update should work
        result = await manager.update_manifest(job.id, {"shot_list": ["shot1", "shot2"]})
        
        assert result is not None
        assert result.production_manifest["shot_list"] == ["shot1", "shot2"]
        
    @pytest.mark.asyncio
    async def test_update_manifest_tracks_history(self, manager):
        """Test that manifest updates are tracked in history."""
        job = await manager.create_job(content="Test")
        await manager.set_manifest(job.id, {"shot_list": [], "duration": 30})
        await manager.update_job_state(job.id, JobState.WAITING_APPROVAL)
        
        await manager.update_manifest(job.id, {"duration": 45})
        
        updated = await manager.get_job(job.id)
        assert updated is not None
        assert len(updated.edits_history) == 1
        assert "duration" in updated.edits_history[0]["changes"]
        
    @pytest.mark.asyncio
    async def test_job_has_timestamps(self, manager):
        """Test that jobs have created_at and updated_at timestamps."""
        job = await manager.create_job(content="Test")
        
        assert job.created_at is not None
        assert job.updated_at is not None
        assert isinstance(job.created_at, datetime)
        
    @pytest.mark.asyncio
    async def test_update_refreshes_timestamp(self, manager):
        """Test that updating a job refreshes updated_at."""
        job = await manager.create_job(content="Test")
        original_updated = job.updated_at
        
        # Small delay to ensure timestamp differs
        await asyncio.sleep(0.01)
        
        await manager.update_job_state(job.id, JobState.ANALYZING)
        
        updated = await manager.get_job(job.id)
        assert updated is not None
        assert updated.updated_at >= original_updated


class TestJobApproval:
    """Tests for job approval workflow."""
    
    @pytest.fixture
    def manager(self):
        return JobManager()
    
    @pytest.mark.asyncio
    async def test_approve_job(self, manager):
        """Test approving a job."""
        job = await manager.create_job(content="Test")
        await manager.set_manifest(job.id, {"title": "Test"})
        await manager.update_job_state(job.id, JobState.WAITING_APPROVAL)
        
        result = await manager.approve_job(job.id)
        
        assert result is not None
        assert result.approved is True
        assert result.approval_timestamp is not None
        assert result.state == JobState.RENDERING
        
    @pytest.mark.asyncio
    async def test_approve_job_wrong_state(self, manager):
        """Test approving job in wrong state."""
        job = await manager.create_job(content="Test")
        # Job is in CREATED state, not WAITING_APPROVAL
        
        result = await manager.approve_job(job.id)
        
        assert result is None
        
    @pytest.mark.asyncio
    async def test_approve_job_nonexistent(self, manager):
        """Test approving non-existent job."""
        result = await manager.approve_job("fake-id")
        assert result is None


class TestJobResults:
    """Tests for job result handling."""
    
    @pytest.fixture
    def manager(self):
        return JobManager()
    
    @pytest.mark.asyncio
    async def test_set_cost_estimate(self, manager):
        """Test setting cost estimate."""
        job = await manager.create_job(content="Test")
        
        cost = {"aiprod_cost": 0.95, "runway_cost": 1.50}
        result = await manager.set_cost_estimate(job.id, cost)
        
        assert result is not None
        assert result.cost_estimate == cost
        
    @pytest.mark.asyncio
    async def test_set_render_result(self, manager):
        """Test setting render result."""
        job = await manager.create_job(content="Test")
        
        render = {"video_url": "https://example.com/video.mp4"}
        result = await manager.set_render_result(job.id, render)
        
        assert result is not None
        assert result.render_result == render
        
    @pytest.mark.asyncio
    async def test_set_qa_report(self, manager):
        """Test setting QA report."""
        job = await manager.create_job(content="Test")
        
        qa = {"overall_score": 0.85, "passed": True}
        result = await manager.set_qa_report(job.id, qa)
        
        assert result is not None
        assert result.qa_report == qa


class TestJobState:
    """Tests for JobState enum."""
    
    def test_all_states_exist(self):
        """Test all expected states are defined."""
        expected = [
            "CREATED", "ANALYZING", "CREATIVE_DIRECTION",
            "WAITING_APPROVAL", "RENDERING", "DELIVERED", "CANCELLED"
        ]
        
        for state_name in expected:
            assert hasattr(JobState, state_name)
            
    def test_state_values(self):
        """Test state values are lowercase strings."""
        assert JobState.CREATED.value == "created"
        assert JobState.RENDERING.value == "rendering"
        assert JobState.DELIVERED.value == "delivered"


class TestICCWorkflow:
    """Integration tests for complete ICC workflow."""
    
    @pytest.fixture
    def manager(self):
        return JobManager()
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, manager):
        """Test complete ICC workflow from creation to approval."""
        # Step 1: Create job
        job = await manager.create_job(
            content="Product demo video for new smartphone",
            preset="brand_campaign"
        )
        job_id = job.id
        
        assert job.state == JobState.CREATED
        
        # Step 2: Move to analyzing
        await manager.update_job_state(job_id, JobState.ANALYZING)
        
        # Step 3: Set manifest (creative direction done)
        manifest = {
            "title": "Smartphone Launch",
            "shot_list": ["intro", "features", "outro"],
            "duration": 60
        }
        await manager.set_manifest(job_id, manifest)
        
        # Step 4: Move to waiting approval
        await manager.update_job_state(job_id, JobState.WAITING_APPROVAL)
        
        # Step 5: Edit manifest (ICC)
        await manager.update_manifest(job_id, {
            "duration": 45,
            "audio_style": "upbeat"
        })
        
        updated = await manager.get_job(job_id)
        assert updated is not None
        assert updated.production_manifest["duration"] == 45
        assert len(updated.edits_history) == 1
        
        # Step 6: Approve for rendering
        approved = await manager.approve_job(job_id)
        
        assert approved is not None
        assert approved.approved is True
        assert approved.state == JobState.RENDERING
        
    @pytest.mark.asyncio
    async def test_multiple_edits_workflow(self, manager):
        """Test making multiple edits before approval."""
        job = await manager.create_job(content="Test")
        await manager.set_manifest(job.id, {"shot_list": [], "duration": 30})
        await manager.update_job_state(job.id, JobState.WAITING_APPROVAL)
        
        # Multiple edits
        await manager.update_manifest(job.id, {"duration": 45})
        await manager.update_manifest(job.id, {"audio_style": "jazz"})
        await manager.update_manifest(job.id, {"duration": 60})
        
        result = await manager.get_job(job.id)
        
        assert result is not None
        # Last edit should win
        assert result.production_manifest["duration"] == 60
        assert result.production_manifest["audio_style"] == "jazz"
        # All edits tracked
        assert len(result.edits_history) == 3
        
    @pytest.mark.asyncio
    async def test_parallel_jobs_workflow(self, manager):
        """Test managing multiple jobs in parallel."""
        jobs = []
        for i in range(3):
            job = await manager.create_job(content=f"Brief {i}")
            jobs.append(job)
            
        # Set manifests for all
        for i, job in enumerate(jobs):
            await manager.set_manifest(job.id, {"title": f"Video {i}"})
            await manager.update_job_state(job.id, JobState.WAITING_APPROVAL)
            
        # Update each differently
        for i, job in enumerate(jobs):
            await manager.update_manifest(job.id, {"duration": 30 + i * 10})
            
        # Verify isolation
        for i, job in enumerate(jobs):
            result = await manager.get_job(job.id)
            assert result is not None
            assert result.production_manifest["title"] == f"Video {i}"
            assert result.production_manifest["duration"] == 30 + i * 10
