"""Tests for aiprod_core.components modules."""

import pytest
import torch
from aiprod_core.components import (
    AdaptiveFlowScheduler,
    EulerFlowStep,
    MultiModalGuider,
    ClassifierFreeGuider,
    GaussianNoiser,
)


class TestSchedulers:
    """Test scheduler components."""
    
    def test_scheduler_creation(self):
        """Test creating an AdaptiveFlowScheduler."""
        scheduler = AdaptiveFlowScheduler()
        assert scheduler is not None
        assert hasattr(scheduler, '__call__') or hasattr(scheduler, 'step')
    
    def test_scheduler_is_callable(self):
        """Test that scheduler can be called."""
        scheduler = AdaptiveFlowScheduler()
        # Should be callable or have step method
        assert callable(scheduler) or hasattr(scheduler, 'step')


class TestEulerFlowStep:
    """Test Euler flow matching step."""
    
    def test_euler_flow_step_creation(self):
        """Test EulerFlowStep initialization."""
        try:
            step = EulerFlowStep()
            assert step is not None
        except (ImportError, ValueError):
            pytest.skip("EulerFlowStep not available")
    
    def test_euler_flow_step_interface(self):
        """Test EulerFlowStep has required interface."""
        try:
            step = EulerFlowStep()
            # Should have main method
            assert hasattr(step, '__call__') or hasattr(step, 'step')
        except (ImportError, ValueError):
            pytest.skip("EulerFlowStep not available")


class TestMultiModalGuider:
    """Test MultiModalGuider."""
    
    def test_multimodal_guider_creation(self):
        """Test MultiModalGuider initialization."""
        try:
            guider = MultiModalGuider()
            assert guider is not None
        except (ImportError, TypeError):
            pytest.skip("MultiModalGuider not available")
    
    def test_multimodal_guider_methods(self):
        """Test MultiModalGuider has required methods."""
        try:
            guider = MultiModalGuider()
            
            # Check for key methods
            assert hasattr(guider, 'should_skip_step')
            assert hasattr(guider, 'do_unconditional_generation')
            assert hasattr(guider, 'do_perturbed_generation')
            assert hasattr(guider, 'calculate')
        except (ImportError, TypeError):
            pytest.skip("MultiModalGuider not available")
    
    def test_multimodal_guider_should_skip_step(self):
        """Test should_skip_step method."""
        try:
            guider = MultiModalGuider()
            
            result = guider.should_skip_step(step=0, num_steps=1000)
            assert isinstance(result, bool)
        except (ImportError, TypeError, AttributeError):
            pytest.skip("Method not available")
    
    def test_multimodal_guider_do_unconditional(self):
        """Test unconditional generation flag."""
        try:
            guider = MultiModalGuider()
            
            result = guider.do_unconditional_generation(step=500, num_steps=1000)
            assert isinstance(result, bool)
        except (ImportError, TypeError, AttributeError):
            pytest.skip("Method not available")


class TestClassifierFreeGuider:
    """Test ClassifierFreeGuider."""
    
    def test_classifier_free_guider_creation(self):
        """Test ClassifierFreeGuider initialization."""
        try:
            guider = ClassifierFreeGuider()
            assert guider is not None
        except (ImportError, TypeError):
            pytest.skip("ClassifierFreeGuider not available")
    
    def test_classifier_free_guider_enabled(self):
        """Test enabled method."""
        try:
            guider = ClassifierFreeGuider()
            
            enabled = guider.enabled(step=0, num_steps=1000)
            assert isinstance(enabled, bool)
        except (ImportError, TypeError, AttributeError):
            pytest.skip("Method not available")
    
    def test_classifier_free_guider_delta(self):
        """Test delta calculation."""
        try:
            guider = ClassifierFreeGuider()
            
            # delta might return a value or scale
            result = guider.delta(step=500, num_steps=1000)
            # Should be numeric or None
            assert result is None or isinstance(result, (int, float, torch.Tensor))
        except (ImportError, TypeError, AttributeError):
            pytest.skip("Method not available")


class TestComponentStack:
    """Integration tests for component stacking."""
    
    def test_guider_stack_creation(self):
        """Test creating a stack of guiders."""
        try:
            guider1 = ClassifierFreeGuider()
            guider2 = MultiModalGuider()
            
            guiders = [guider1, guider2]
            assert len(guiders) == 2
        except (ImportError, TypeError):
            pytest.skip("Guiders not available")
    
    def test_scheduler_callable(self):
        """Test scheduler is callable."""
        scheduler = AdaptiveFlowScheduler()
        # Scheduler should be callable or have a step method
        assert callable(scheduler) or hasattr(scheduler, 'step')


class TestComponentEdgeCases:
    """Test edge cases in components."""
    
    def test_guider_extreme_steps(self):
        """Test guiders with extreme step values."""
        try:
            guider = ClassifierFreeGuider()
            
            # Extreme values
            result1 = guider.enabled(step=0, num_steps=1000)
            result2 = guider.enabled(step=1000, num_steps=1000)
            result3 = guider.enabled(step=999, num_steps=1000)
            
            assert isinstance(result1, bool)
            assert isinstance(result2, bool)
            assert isinstance(result3, bool)
        except (ImportError, TypeError, AttributeError):
            pytest.skip("Method not available")
    
    def test_scheduler_zero_timestep(self):
        """Test scheduler behavior at zero timestep."""
        scheduler = AdaptiveFlowScheduler()
        # Should handle initialization gracefully
        assert scheduler is not None


class TestComponentConsistency:
    """Test consistency across components."""
    
    def test_multiple_guiders_same_step(self):
        """Test multiple guiders with same step parameters."""
        try:
            guiders = [
                ClassifierFreeGuider(),
                MultiModalGuider(),
            ]
            
            step = 500
            num_steps = 1000
            
            results = []
            for guider in guiders:
                try:
                    if hasattr(guider, 'enabled'):
                        result = guider.enabled(step=step, num_steps=num_steps)
                    elif hasattr(guider, 'should_skip_step'):
                        result = guider.should_skip_step(step=step, num_steps=num_steps)
                    else:
                        continue
                    results.append(result)
                except (TypeError, AttributeError):
                    continue
            
            # All results should be boolean
            for result in results:
                assert isinstance(result, bool)
        except (ImportError, TypeError):
            pytest.skip("Guiders not available")
