"""Tests for reward modeling system"""

import pytest
import torch
import numpy as np

from aiprod_pipelines.inference.reward_modeling import (
    RewardModelForAutoTuning,
    RewardNet,
    UserFeedback,
    UserProfile,
    ABTestingFramework,
)


class TestRewardNet:
    """Test reward network"""
    
    def test_reward_net_initialization(self):
        """Test reward network creation"""
        net = RewardNet(embedding_dim=512, hidden_dim=256)
        assert net is not None
    
    def test_reward_net_forward_pass(self):
        """Test forward pass through network"""
        net = RewardNet()
        
        batch_size = 4
        text_emb = torch.randn(batch_size, 512)
        video_emb = torch.randn(batch_size, 512)
        cfg = torch.randn(batch_size)
        stg = torch.randn(batch_size)
        steps = torch.randn(batch_size)
        
        output = net(text_emb, video_emb, cfg, stg, steps)
        
        assert output.shape == (batch_size, 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output


class TestUserFeedback:
    """Test user feedback"""
    
    def test_feedback_creation(self):
        """Test feedback object"""
        feedback = UserFeedback(
            feedback_id="fb_001",
            user_id="user_001",
            generation_id="gen_001",
            prompt="a beautiful sunset",
            quality_rating=0.9,
            speed_rating=0.7,
            aesthetics_rating=0.85,
        )
        
        assert feedback.feedback_id == "fb_001"
        assert feedback.quality_rating == 0.9
        assert abs(feedback.overall_rating - (0.9 + 0.7 + 0.85) / 3.0) < 1e-6


class TestUserProfile:
    """Test user profiles"""
    
    def test_profile_initialization(self):
        """Test profile creation"""
        profile = UserProfile(user_id="user_001")
        
        assert profile.user_id == "user_001"
        assert len(profile.feedback_history) == 0
        assert profile.average_quality_rating == 0.5
    
    def test_profile_update_from_feedback(self):
        """Test profile update"""
        profile = UserProfile(user_id="user_001")
        
        feedback1 = UserFeedback(
            feedback_id="fb_001",
            user_id="user_001",
            generation_id="gen_001",
            prompt="test",
            quality_rating=1.0,
            speed_rating=1.0,
            aesthetics_rating=1.0,
            cfg_scale=8.0,
            stg_scale=4.5,
            num_steps=55,
        )
        
        profile.update_from_feedback(feedback1)
        
        assert len(profile.feedback_history) == 1
        assert profile.average_quality_rating == 1.0
        assert profile.preferred_cfg_range[1] == 8.0
        assert profile.preferred_stg_range[1] == 4.5


class TestRewardModelForAutoTuning:
    """Test reward model system"""
    
    @pytest.fixture
    def model(self):
        """Create reward model"""
        return RewardModelForAutoTuning(device="cpu")
    
    def test_model_initialization(self, model):
        """Test initialization"""
        assert model is not None
        assert model.device == "cpu"
        assert len(model.user_profiles) == 0
    
    @pytest.mark.asyncio
    async def test_collect_feedback(self, model):
        """Test feedback collection"""
        result = await model.collect_user_feedback(
            user_id="user_001",
            generation_id="gen_001",
            prompt="test prompt",
            quality_rating=0.8,
            speed_rating=0.7,
            aesthetics_rating=0.75,
            cfg_scale=7.5,
            stg_scale=4.0,
        )
        
        assert result["accepted"] is True
        assert "feedback_id" in result
        assert result["buffer_size"] == 1
        assert "user_001" in model.user_profiles
    
    @pytest.mark.asyncio
    async def test_suggest_hyperparameters_default(self, model):
        """Test hyperparameter suggestion for new user"""
        suggestions = await model.suggest_hyperparameters(
            user_id="new_user",
            prompt="test",
            priority="balanced",
        )
        
        assert "cfg_scale" in suggestions
        assert "stg_scale" in suggestions
        assert "num_steps" in suggestions
        assert suggestions["priority"] == "balanced"
        assert suggestions["based_on_profile"] is False
    
    @pytest.mark.asyncio
    async def test_suggest_hyperparameters_quality_priority(self, model):
        """Test quality-focused suggestion"""
        suggestions = await model.suggest_hyperparameters(
            user_id="user_001",
            prompt="test",
            priority="quality",
        )
        
        assert "cfg_scale" in suggestions
        assert suggestions["priority"] == "quality"
    
    @pytest.mark.asyncio
    async def test_suggest_hyperparameters_speed_priority(self, model):
        """Test speed-focused suggestion"""
        suggestions = await model.suggest_hyperparameters(
            user_id="user_001",
            prompt="test",
            priority="speed",
        )
        
        assert "cfg_scale" in suggestions
        assert suggestions["priority"] == "speed"
    
    def test_user_profile_summary(self, model):
        """Test profile summary"""
        # Non-existent user
        summary = model.get_user_profile_summary("nonexistent")
        assert summary["status"] == "no_profile"
        
        # Create profile
        profile = UserProfile(user_id="user_001")
        model.user_profiles["user_001"] = profile
        
        summary = model.get_user_profile_summary("user_001")
        assert summary["user_id"] == "user_001"
        assert summary["feedback_count"] == 0


class TestABTestingFramework:
    """Test A/B testing"""
    
    @pytest.fixture
    def framework(self):
        """Create testing framework"""
        return ABTestingFramework()
    
    @pytest.mark.asyncio
    async def test_start_test(self, framework):
        """Test starting A/B test"""
        config = await framework.start_ab_test(
            test_id="test_001",
            variant_a={"cfg_scale": 7.5},
            variant_b={"cfg_scale": 9.0},
            sample_size=50,
        )
        
        assert config.test_id == "test_001"
        assert "test_001" in framework.active_tests
    
    @pytest.mark.asyncio
    async def test_record_results(self, framework):
        """Test recording results"""
        await framework.start_ab_test(
            test_id="test_001",
            variant_a={"cfg_scale": 7.5},
            variant_b={"cfg_scale": 9.0},
        )
        
        # Record variant A results
        for i in range(10):
            await framework.record_result("test_001", "A", 0.7 + i * 0.01)
        
        # Record variant B results
        for i in range(10):
            await framework.record_result("test_001", "B", 0.8 + i * 0.01)
        
        summary = await framework.get_test_summary("test_001")
        assert summary["variant_a"]["sample_size"] == 10
        assert summary["variant_b"]["sample_size"] == 10
    
    @pytest.mark.asyncio
    async def test_determine_winner(self, framework):
        """Test winner determination"""
        await framework.start_ab_test(
            test_id="test_001",
            variant_a={"cfg_scale": 7.5},
            variant_b={"cfg_scale": 9.0},
        )
        
        # Make variant B clearly better
        for i in range(5):
            await framework.record_result("test_001", "A", 0.5)
            await framework.record_result("test_001", "B", 0.9)
        
        summary = await framework.get_test_summary("test_001")
        assert summary["winner"] == "B"
    
    @pytest.mark.asyncio
    async def test_complete_test(self, framework):
        """Test completing test"""
        await framework.start_ab_test(
            test_id="test_001",
            variant_a={"cfg_scale": 7.5},
            variant_b={"cfg_scale": 9.0},
        )
        
        result = await framework.complete_test("test_001")
        
        assert "test_001" not in framework.active_tests
        assert len(framework.completed_tests) == 1
    
    def test_history(self, framework):
        """Test test history"""
        history = framework.get_test_history()
        assert isinstance(history, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
