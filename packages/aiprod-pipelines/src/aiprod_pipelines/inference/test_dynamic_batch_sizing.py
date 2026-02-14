"""
Comprehensive Test Suite for Dynamic Batch Sizing

Tests all components of the dynamic batch sizing infrastructure.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dynamic_batch_sizing import (
    BatchSizingStrategy, MemoryManagementMode, DeviceType,
    BatchSizingConstraints, BatchSizingConfig, BatchSizeDecision,
    LinearBatchSizer, ExponentialBatchSizer, PerformanceProfile,
    MemoryAllocationPattern, MemoryProfile, GPUMemoryMonitor,
    MemoryFragmentationEstimator, ComputeMemoryTradeoffAnalyzer,
    BatchPerformanceMetrics, AdaptiveBatcherConfig, AdaptiveBatcher,
    PerformancePredictor, DynamicBatchOptimizer,
    CacheKey, CacheEntry, BatchSizingCache, ModelProfileRegistry,
    PerformanceModel, PerformanceCurveParams, CurveEstimator,
    PerformanceProfiler, ResourceEstimator, PerformanceMonitor,
    DynamicBatchConfig, DynamicBatchMetrics, DynamicBatchSizer,
    MultiModelBatchOptimizer
)


class TestBatchSizingStrategy(unittest.TestCase):
    """Test batch sizing strategy enumeration and constraints."""
    
    def test_batch_sizing_strategies_exist(self):
        """Test that all strategies are defined."""
        self.assertTrue(hasattr(BatchSizingStrategy, 'FIXED'))
        self.assertTrue(hasattr(BatchSizingStrategy, 'ADAPTIVE'))
        self.assertEqual(BatchSizingStrategy.FIXED.value, "fixed")
    
    def test_memory_management_modes(self):
        """Test memory management modes."""
        self.assertEqual(len(MemoryManagementMode.__members__), 4)
        self.assertIn('BALANCED', MemoryManagementMode.__members__)
    
    def test_device_types(self):
        """Test device type enumeration."""
        self.assertTrue(hasattr(DeviceType, 'GPU_HIGH_END'))
        self.assertTrue(hasattr(DeviceType, 'GPU_MOBILE'))
    
    def test_constraints_validation(self):
        """Test batch sizing constraints validation."""
        constraints = BatchSizingConstraints()
        self.assertTrue(constraints.validate())
        
        # Invalid: min > max
        bad_constraints = BatchSizingConstraints(min_batch_size=100, max_batch_size=50)
        self.assertFalse(bad_constraints.validate())
    
    def test_constraints_device_adjustment(self):
        """Test device-based constraint adjustment."""
        constraints = BatchSizingConstraints()
        constraints.adjust_for_device(DeviceType.GPU_HIGH_END)
        self.assertEqual(constraints.memory_limit_mb, 80000.0)
        self.assertEqual(constraints.max_batch_size, 512)
        
        constraints.adjust_for_device(DeviceType.GPU_MOBILE)
        self.assertEqual(constraints.memory_limit_mb, 6000.0)


class TestLinearBatchSizer(unittest.TestCase):
    """Test linear batch sizing strategy."""
    
    def test_linear_sizer_initialization(self):
        """Test linear sizer creation."""
        config = BatchSizingConfig()
        sizer = LinearBatchSizer(config)
        self.assertIsNotNone(sizer)
    
    def test_linear_batch_proposal(self):
        """Test batch size proposal."""
        config = BatchSizingConfig(
            constraints=BatchSizingConstraints(memory_limit_mb=10000.0)
        )
        sizer = LinearBatchSizer(config)
        
        decision = sizer.propose_batch_size(
            memory_per_sample_mb=10.0,
            latency_per_sample_ms=1.0,
            throughput_per_sample=1.0,
        )
        
        self.assertIsNotNone(decision)
        self.assertGreater(decision.batch_size, 0)
        self.assertEqual(decision.strategy_used, BatchSizingStrategy.LINEAR)


class TestMemoryProfiler(unittest.TestCase):
    """Test memory profiling capabilities."""
    
    def test_memory_profile_creation(self):
        """Test memory profile creation."""
        profile = MemoryProfile(
            model_name="test_model",
            device_type="gpu",
            base_memory_mb=100.0,
            memory_per_sample_mb=10.0,
        )
        self.assertEqual(profile.model_name, "test_model")
    
    def test_memory_estimation(self):
        """Test memory estimation."""
        profile = MemoryProfile(
            model_name="test",
            device_type="gpu",
            base_memory_mb=100.0,
            memory_per_sample_mb=10.0,
        )
        
        estimated = profile.estimate_memory(batch_size=32)
        expected = 100.0 + 10.0 * 32 * 1.2  # With peak multiplier
        self.assertAlmostEqual(estimated, expected, places=1)
    
    def test_batch_recommendation(self):
        """Test batch size recommendation from memory."""
        profile = MemoryProfile(
            model_name="test",
            device_type="gpu",
            base_memory_mb=50.0,
            memory_per_sample_mb=5.0,
        )
        
        batch_size = profile.recommend_batch_for_memory(target_memory_mb=200.0)
        self.assertGreater(batch_size, 0)


class TestAdaptiveBatcher(unittest.TestCase):
    """Test adaptive batch sizing."""
    
    def test_adaptive_batcher_creation(self):
        """Test adaptive batcher initialization."""
        config = AdaptiveBatcherConfig(initial_batch_size=32)
        batcher = AdaptiveBatcher(config)
        self.assertEqual(batcher.current_batch_size, 32)
    
    def test_record_measurement(self):
        """Test recording performance metrics."""
        config = AdaptiveBatcherConfig()
        batcher = AdaptiveBatcher(config)
        
        metrics = BatchPerformanceMetrics(
            batch_size=32,
            latency_ms=100.0,
            throughput_samples_per_sec=320.0,
            memory_peak_mb=1000.0,
            memory_avg_mb=800.0,
        )
        
        batcher.record_measurement(metrics)
        self.assertEqual(len(batcher.metrics_history), 1)
    
    def test_stability_tracking(self):
        """Test batch stability tracking."""
        config = AdaptiveBatcherConfig(min_stable_iterations=3)
        batcher = AdaptiveBatcher(config)
        
        # Record stable measurements
        for i in range(3):
            metrics = BatchPerformanceMetrics(
                batch_size=32,
                latency_ms=100.0,
                throughput_samples_per_sec=320.0,
                memory_peak_mb=1000.0,
                memory_avg_mb=800.0,
                success=True,
            )
            batcher.record_measurement(metrics)
        
        confidence = batcher.get_stability_confidence()
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)


class TestPerformancePredictor(unittest.TestCase):
    """Test performance prediction."""
    
    def test_predictor_initialization(self):
        """Test predictor creation."""
        predictor = PerformancePredictor()
        self.assertIsNotNone(predictor)
    
    def test_predictor_calibration(self):
        """Test predictor calibration."""
        predictor = PerformancePredictor()
        
        measurements = [
            BatchPerformanceMetrics(batch_size=16, latency_ms=50.0, throughput_samples_per_sec=320.0,
                                   memory_peak_mb=500.0, memory_avg_mb=400.0),
            BatchPerformanceMetrics(batch_size=32, latency_ms=100.0, throughput_samples_per_sec=320.0,
                                   memory_peak_mb=1000.0, memory_avg_mb=800.0),
        ]
        
        predictor.calibrate(measurements)
        self.assertEqual(len(predictor.calibration_points), 2)
    
    def test_latency_prediction(self):
        """Test latency prediction."""
        predictor = PerformancePredictor()
        
        measurements = [
            BatchPerformanceMetrics(batch_size=16, latency_ms=50.0, throughput_samples_per_sec=320.0,
                                   memory_peak_mb=500.0, memory_avg_mb=400.0),
            BatchPerformanceMetrics(batch_size=32, latency_ms=100.0, throughput_samples_per_sec=320.0,
                                   memory_peak_mb=1000.0, memory_avg_mb=800.0),
        ]
        
        predictor.calibrate(measurements)
        predicted = predictor.predict_latency(batch_size=24)
        self.assertGreater(predicted, 0)


class TestBatchSizingCache(unittest.TestCase):
    """Test batch sizing cache."""
    
    def test_cache_put_get(self):
        """Test caching entries."""
        cache = BatchSizingCache(max_entries=10)
        
        key = CacheKey(
            model_name="test",
            model_hash="abc123",
            device_type="gpu",
            device_compute_capability="8.0",
            memory_available_mb=40000,
            target_optimization="throughput"
        )
        
        entry = CacheEntry(
            key=key,
            batch_size=32,
            predicted_latency_ms=100.0,
            predicted_memory_mb=1000.0,
            predicted_throughput=320.0,
            confidence_score=0.95,
            timestamp=__import__('time').time(),
        )
        
        cache.put(entry)
        retrieved = cache.get(key)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.batch_size, 32)
    
    def test_cache_lru_eviction(self):
        """Test LRU eviction."""
        cache = BatchSizingCache(max_entries=2)
        
        for i in range(3):
            key = CacheKey(
                model_name=f"model{i}",
                model_hash=f"hash{i}",
                device_type="gpu",
                device_compute_capability="8.0",
                memory_available_mb=40000,
                target_optimization="throughput"
            )
            
            entry = CacheEntry(
                key=key,
                batch_size=32,
                predicted_latency_ms=100.0,
                predicted_memory_mb=1000.0,
                predicted_throughput=320.0,
                confidence_score=0.95,
                timestamp=__import__('time').time(),
            )
            cache.put(entry)
        
        self.assertLessEqual(len(cache.cache), 2)


class TestDynamicBatchSizer(unittest.TestCase):
    """Test main dynamic batch sizer."""
    
    def test_sizer_initialization(self):
        """Test sizer initialization."""
        config = DynamicBatchConfig()
        sizer = DynamicBatchSizer(config)
        self.assertIsNotNone(sizer)
    
    def test_get_suggested_batch_size(self):
        """Test getting suggested batch size."""
        config = DynamicBatchConfig()
        sizer = DynamicBatchSizer(config)
        sizer.initialize("test_model", "gpu", 40000.0)
        
        batch_size = sizer.suggest_batch_size()
        self.assertGreater(batch_size, 0)
    
    def test_record_performance(self):
        """Test recording batch performance."""
        config = DynamicBatchConfig()
        sizer = DynamicBatchSizer(config)
        sizer.initialize("test_model", "gpu", 40000.0)
        
        sizer.record_batch_performance(
            batch_size=32,
            latency_ms=100.0,
            memory_mb=1000.0,
            throughput_samples_per_sec=320.0,
            gpu_utilization_percent=85.0,
            success=True,
        )
        
        self.assertEqual(sizer.metrics.iterations, 1)


class TestMultiModelBatchOptimizer(unittest.TestCase):
    """Test multi-model batch optimization."""
    
    def test_register_model(self):
        """Test registering multiple models."""
        config = DynamicBatchConfig()
        optimizer = MultiModelBatchOptimizer(config)
        
        sizer1 = optimizer.register_model("model1", "gpu", 40000.0)
        sizer2 = optimizer.register_model("model2", "gpu", 20000.0)
        
        self.assertIsNotNone(sizer1)
        self.assertIsNotNone(sizer2)
        self.assertNotEqual(sizer1, sizer2)
    
    def test_get_all_metrics(self):
        """Test getting metrics for all models."""
        config = DynamicBatchConfig()
        optimizer = MultiModelBatchOptimizer(config)
        
        optimizer.register_model("model1", "gpu", 40000.0)
        optimizer.register_model("model2", "gpu", 20000.0)
        
        metrics = optimizer.get_all_metrics()
        self.assertEqual(len(metrics), 2)
        self.assertIn("model1", metrics)
        self.assertIn("model2", metrics)


class TestIntegration(unittest.TestCase):
    """Integration tests for batch sizing system."""
    
    def test_end_to_end_batch_sizing(self):
        """Test end-to-end batch sizing workflow."""
        config = DynamicBatchConfig(
            enable_profiling=True,
            enable_adaptation=True,
            enable_caching=True,
        )
        
        sizer = DynamicBatchSizer(config)
        sizer.initialize("test_model", "gpu", 40000.0)
        
        # Simulate inference loop
        for i in range(20):
            batch_size = sizer.suggest_batch_size()
            self.assertGreater(batch_size, 0)
            
            sizer.record_batch_performance(
                batch_size=batch_size,
                latency_ms=100.0 + i * 0.5,
                memory_mb=1000.0,
                throughput_samples_per_sec=320.0,
                gpu_utilization_percent=85.0,
                success=True,
            )
        
        metrics = sizer.get_metrics()
        self.assertEqual(metrics.iterations, 20)


if __name__ == '__main__':
    unittest.main()
