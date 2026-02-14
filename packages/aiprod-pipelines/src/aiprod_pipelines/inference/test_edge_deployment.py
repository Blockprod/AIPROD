"""
Comprehensive Test Suite for Edge Deployment

Tests all components of the edge deployment infrastructure.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from edge_deployment import (
    # Model Optimization
    EdgeTargetDevice, OptimizationObjective, EdgeDeviceProfile,
    OptimizationConfig, ModelCompressionMetrics, LayerOptimizer,
    # Quantization
    QuantizationType, QuantizationLevel, QuantizationStats, QuantizationScheme,
    PostTrainingQuantizer, MixedPrecisionQuantizer,
    # Pruning
    PruningStrategy, PruningCriterion, WeightPruner, StructuredPruner,
    IterativePruner, KnowledgeDistillation,
    # Mobile Runtime
    MobileRuntime, RuntimeMemoryConfig, RuntimeConfig,
    InferenceMemoryManager, MobileInferenceRuntime,
    # Edge Inference
    EdgeInferenceMetrics, InferenceCache, EdgeInferenceEngine,
    BatchedEdgeInference,
    # Resource Monitoring
    ResourceLevel, DeviceResourceMonitor, AdaptiveInferenceController,
    # Deployment
    DeploymentStatus, EdgeDeploymentManager
)


class TestEdgeTargetDevices(unittest.TestCase):
    """Test edge target device enumeration."""
    
    def test_device_types_exist(self):
        """Test that edge device types are defined."""
        self.assertTrue(hasattr(EdgeTargetDevice, 'MOBILE_PHONE'))
        self.assertTrue(hasattr(EdgeTargetDevice, 'EDGE_SERVER'))
        self.assertTrue(hasattr(EdgeTargetDevice, 'JETSON_NANO'))
    
    def test_optimization_objectives(self):
        """Test optimization objective enumeration."""
        self.assertTrue(hasattr(OptimizationObjective, 'LATENCY'))
        self.assertTrue(hasattr(OptimizationObjective, 'MEMORY'))
        self.assertTrue(hasattr(OptimizationObjective, 'BATTERY'))


class TestEdgeDeviceProfile(unittest.TestCase):
    """Test edge device profiling."""
    
    def test_device_profile_creation(self):
        """Test device profile creation."""
        profile = EdgeDeviceProfile(
            device_type=EdgeTargetDevice.MOBILE_PHONE,
            device_name="iPhone 14",
            cpu_cores=6,
            ram_mb=6000,
            storage_mb=128000,
        )
        self.assertEqual(profile.device_name, "iPhone 14")
        self.assertEqual(profile.total_memory_mb, 6000)
    
    def test_memory_budget_calculation(self):
        """Test memory budget calculation."""
        profile = EdgeDeviceProfile(
            device_type=EdgeTargetDevice.MOBILE_PHONE,
            device_name="Phone",
            cpu_cores=4,
            ram_mb=4000,
            storage_mb=64000,
        )
        self.assertEqual(profile.total_memory_mb, 4000)
        self.assertEqual(profile.memory_budget_mb, int(4000 * 0.8))


class TestModelCompressionStrategies(unittest.TestCase):
    """Test model compression strategies."""
    
    def test_compression_config(self):
        """Test compression configuration."""
        profile = EdgeDeviceProfile(
            device_type=EdgeTargetDevice.JETSON_NANO,
            device_name="Jetson",
            cpu_cores=4,
            ram_mb=4000,
            storage_mb=16000,
        )
        
        config = OptimizationConfig(
            target_device=EdgeTargetDevice.JETSON_NANO,
            device_profile=profile,
            enable_quantization=True,
            enable_pruning=True,
        )
        
        self.assertTrue(config.enable_quantization)
        self.assertEqual(config.quantization_bits, 8)
    
    def test_layer_importance_computation(self):
        """Test layer importance computation."""
        from edge_deployment import LayerOptimizer
        
        importance = LayerOptimizer.recommend_pruning_percentage(
            layer_importance=0.8,
            target_speedup=2.0,
        )
        
        self.assertGreater(importance, 0.0)
        self.assertLess(importance, 1.0)


class TestQuantization(unittest.TestCase):
    """Test quantization techniques."""
    
    def test_quantization_types(self):
        """Test quantization type enumeration."""
        self.assertTrue(hasattr(QuantizationType, 'SYMMETRIC'))
        self.assertTrue(hasattr(QuantizationType, 'ASYMMETRIC'))
    
    def test_quantization_levels(self):
        """Test quantization levels."""
        self.assertEqual(QuantizationLevel.INT4.value, 4)
        self.assertEqual(QuantizationLevel.INT8.value, 8)
    
    def test_post_training_quantizer(self):
        """Test post-training quantization."""
        quantizer = PostTrainingQuantizer(calibration_samples=100)
        self.assertIsNotNone(quantizer)
    
    def test_mixed_precision_quantizer(self):
        """Test mixed-precision quantization."""
        quantizer = MixedPrecisionQuantizer()
        
        model_structure = {
            "layer1": {"importance": 0.9},
            "layer2": {"importance": 0.5},
            "layer3": {"importance": 0.2},
        }
        
        precision_map = quantizer.select_precision_per_layer(
            model_structure,
            target_model_size_mb=50.0,
            current_model_size_mb=200.0,
        )
        
        self.assertEqual(len(precision_map), 3)


class TestPruning(unittest.TestCase):
    """Test pruning techniques."""
    
    def test_pruning_strategies(self):
        """Test pruning strategy enumeration."""
        self.assertTrue(hasattr(PruningStrategy, 'UNSTRUCTURED'))
        self.assertTrue(hasattr(PruningStrategy, 'STRUCTURED'))
        self.assertTrue(hasattr(PruningStrategy, 'ITERATIVE'))
    
    def test_weight_pruner(self):
        """Test weight pruning."""
        pruner = WeightPruner(PruningStrategy.UNSTRUCTURED)
        
        weights = [1.0, 0.1, 5.0, 0.01, 2.0]
        mask = pruner.generate_pruning_mask(
            weights=weights,
            pruning_ratio=0.4,
        )
        
        self.assertEqual(len(mask.mask), len(weights))
        self.assertEqual(mask.pruned_count, 2)
    
    def test_structured_pruner(self):
        """Test structured pruning."""
        pruner = StructuredPruner()
        
        filters = [
            [1.0, 0.5, 2.0],
            [0.1, 0.05, 0.2],
            [5.0, 2.5, 10.0],
        ]
        
        importance = pruner.compute_channel_importance("layer1", filters)
        self.assertEqual(len(importance), 3)
    
    def test_iterative_pruner(self):
        """Test iterative pruning."""
        pruner = IterativePruner(max_iterations=5)
        
        result = pruner.iterative_prune(
            initial_model_size_mb=100.0,
            target_model_size_mb=25.0,
            eval_function=lambda x: 0.95,
        )
        
        self.assertIn("final_size_mb", result)
        self.assertLess(result["final_size_mb"], 100.0)


class TestKnowledgeDistillation(unittest.TestCase):
    """Test knowledge distillation."""
    
    def test_distillation_initialization(self):
        """Test distillation initialization."""
        distiller = KnowledgeDistillation(temperature=4.0)
        self.assertEqual(distiller.temperature, 4.0)
    
    def test_student_performance_estimation(self):
        """Test student model performance estimation."""
        distiller = KnowledgeDistillation()
        
        student_acc = distiller.estimate_student_performance(
            teacher_accuracy=1.0,
            distillation_iterations=100,
        )
        
        self.assertGreater(student_acc, 0.9)
        self.assertLessEqual(student_acc, 1.0)


class TestMobileRuntime(unittest.TestCase):
    """Test mobile runtime infrastructure."""
    
    def test_memory_manager(self):
        """Test inference memory manager."""
        config = RuntimeMemoryConfig(max_memory_mb=512)
        manager = InferenceMemoryManager(config)
        
        success = manager.allocate(100.0)
        self.assertTrue(success)
        
        util = manager.get_utilization_percent()
        self.assertAlmostEqual(util, 19.53, places=1)  # 100/512 * 100
    
    def test_memory_exhaustion(self):
        """Test memory exhaustion handling."""
        config = RuntimeMemoryConfig(max_memory_mb=100)
        manager = InferenceMemoryManager(config)
        
        success1 = manager.allocate(60.0)
        self.assertTrue(success1)
        
        success2 = manager.allocate(50.0)  # More than remaining
        self.assertFalse(success2)


class TestEdgeInference(unittest.TestCase):
    """Test edge inference engine."""
    
    def test_inference_cache(self):
        """Test inference caching."""
        cache = InferenceCache(max_entries=10)
        
        cache.put("input1", {"result": [1, 2, 3]})
        cached = cache.get("input1")
        
        self.assertIsNotNone(cached)
        self.assertEqual(cached["result"], [1, 2, 3])
    
    def test_cache_eviction(self):
        """Test cache eviction on max entries."""
        cache = InferenceCache(max_entries=2)
        
        cache.put("input1", {"result": 1})
        cache.put("input2", {"result": 2})
        cache.put("input3", {"result": 3})
        
        self.assertEqual(len(cache.cache), 2)


class TestResourceMonitoring(unittest.TestCase):
    """Test device resource monitoring."""
    
    def test_resource_monitor(self):
        """Test resource monitoring."""
        monitor = DeviceResourceMonitor()
        snapshot = monitor.collect_snapshot()
        
        self.assertIsNotNone(snapshot)
        self.assertGreaterEqual(snapshot.cpu_percent, 0.0)
        self.assertLessEqual(snapshot.cpu_percent, 100.0)
    
    def test_resource_levels(self):
        """Test resource level determination."""
        monitor = DeviceResourceMonitor()
        levels = monitor.get_resource_levels()
        
        self.assertIn("cpu", levels)
        self.assertIn("memory", levels)
        self.assertIsInstance(levels["cpu"], ResourceLevel)
    
    def test_adaptive_inference_controller(self):
        """Test adaptive inference controller."""
        monitor = DeviceResourceMonitor()
        controller = AdaptiveInferenceController(monitor)
        
        batch_size = controller.get_recommended_batch_size()
        self.assertGreater(batch_size, 0)
        
        should_defer = controller.should_defer_inference()
        self.assertIsInstance(should_defer, bool)


class TestEdgeDeployment(unittest.TestCase):
    """Test deployment management."""
    
    def test_deployment_manager_initialization(self):
        """Test deployment manager creation."""
        manager = EdgeDeploymentManager()
        self.assertIsNotNone(manager)
    
    def test_create_deployment_package(self):
        """Test creating deployment package."""
        manager = EdgeDeploymentManager()
        
        package = manager.create_deployment_package(
            model_name="test_model",
            model_version="1.0",
            model_path="/path/to/model",
            target_device="mobile_phone",
        )
        
        self.assertIsNotNone(package)
        self.assertEqual(package.model_name, "test_model")
    
    def test_deploy_to_device(self):
        """Test deploying to device."""
        manager = EdgeDeploymentManager()
        
        package = manager.create_deployment_package(
            model_name="test_model",
            model_version="1.0",
            model_path="/path/to/model",
            target_device="mobile_phone",
        )
        
        record = manager.deploy_to_device(
            package=package,
            device_id="device_123",
        )
        
        self.assertTrue(record.success)
        self.assertEqual(record.deployment_status, DeploymentStatus.ACTIVE)
    
    def test_get_active_model(self):
        """Test getting active model on device."""
        manager = EdgeDeploymentManager()
        
        package = manager.create_deployment_package(
            model_name="test_model",
            model_version="1.0",
            model_path="/path/to/model",
            target_device="mobile_phone",
        )
        
        manager.deploy_to_device(package, "device_123")
        active = manager.get_active_model("device_123")
        
        self.assertIsNotNone(active)
        self.assertEqual(active["model"], "test_model")


class TestIntegration(unittest.TestCase):
    """Integration tests for edge deployment."""
    
    def test_end_to_end_optimization_and_deployment(self):
        """Test end-to-end optimization and deployment workflow."""
        # Device profile
        profile = EdgeDeviceProfile(
            device_type=EdgeTargetDevice.MOBILE_PHONE,
            device_name="iPhone 14",
            cpu_cores=6,
            ram_mb=6000,
            storage_mb=128000,
        )
        
        # Optimization config
        config = OptimizationConfig(
            target_device=EdgeTargetDevice.MOBILE_PHONE,
            device_profile=profile,
            enable_quantization=True,
            enable_pruning=True,
        )
        
        # Deployment
        manager = EdgeDeploymentManager()
        package = manager.create_deployment_package(
            model_name="optimized_model",
            model_version="1.0",
            model_path="/path/to/model",
            target_device="mobile_phone",
            optimizations={"quantization": True, "pruning": True},
        )
        
        record = manager.deploy_to_device(package, "iphone_001")
        
        self.assertTrue(record.success)


if __name__ == '__main__':
    unittest.main()
