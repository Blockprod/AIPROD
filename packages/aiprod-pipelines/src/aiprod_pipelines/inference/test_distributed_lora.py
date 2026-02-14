"""
Distributed LoRA Tests

Comprehensive test suite for distributed LoRA fine-tuning, federated learning,
model management, and registry operations.
"""

import pytest
from datetime import datetime
from typing import Dict, Any


class TestDistributedLoRAConfig:
    """Tests for LoRA configuration"""
    
    def test_lorarankpreset(self):
        from aiprod_pipelines.inference.distributed_lora import LoRARank
        
        assert LoRARank.SMALL.value == (4, 8)
        assert LoRARank.MEDIUM.value == (8, 16)
        assert LoRARank.LARGE.value == (16, 32)
    
    def test_distributed_lora_config(self):
        from aiprod_pipelines.inference.distributed_lora import DistributedLoRAConfig, LoRARank
        
        config = DistributedLoRAConfig(
            rank=LoRARank.MEDIUM,
            learning_rate=1e-4,
            max_steps=5000
        )
        
        assert config.learning_rate == 1e-4
        r, alpha = config.get_lora_r_alpha()
        assert r == 8 and alpha == 16
    
    def test_user_lora_preset(self):
        from aiprod_pipelines.inference.distributed_lora import UserLoRAPreset, LoRARank, LoRATarget
        
        preset = UserLoRAPreset(
            preset_id="test_preset",
            name="Test",
            description="Test preset",
            rank=LoRARank.MEDIUM,
            target_modules=LoRATarget.ATTENTION_QKV,
            learning_rate=5e-5,
            training_steps=3000
        )
        
        config = preset.to_config()
        assert config.learning_rate == 5e-5
    
    def test_lora_model_metadata(self):
        from aiprod_pipelines.inference.distributed_lora import LoRAModelMetadata
        
        metadata = LoRAModelMetadata(
            model_id="model_001",
            tenant_id="tenant_1",
            user_id="user_1",
            base_model="base_model",
            rank=8,
            alpha=16,
            num_parameters=1_000_000,
            training_steps=5000,
            training_loss=0.5
        )
        
        data_dict = metadata.to_dict()
        assert data_dict["model_id"] == "model_001"
        assert data_dict["rank"] == 8
    
    def test_config_manager(self):
        from aiprod_pipelines.inference.distributed_lora import DistributedLoRAConfigManager
        
        manager = DistributedLoRAConfigManager()
        
        # Test default presets
        presets = manager.list_presets()
        assert len(presets) > 0
        
        # Get preset
        quick_preset = manager.get_preset("quick")
        assert quick_preset is not None
        
        # Create config from preset
        config = manager.create_config_from_preset("balanced")
        assert config is not None


class TestFederatedLearning:
    """Tests for federated training"""
    
    def test_client_update(self):
        from aiprod_pipelines.inference.distributed_lora import ClientUpdate
        
        update = ClientUpdate(
            client_id="client_1",
            model_weights={"layer1": [1.0, 2.0]},
            weight_deltas={"layer1": [0.1, 0.2]},
            num_samples_trained=1000,
            training_steps=50,
            loss=0.5
        )
        
        assert update.client_id == "client_1"
        norm = update.get_update_norm()
        assert norm > 0
    
    def test_federated_aggregator(self):
        from aiprod_pipelines.inference.distributed_lora import FederatedAggregator, ClientUpdate, FederatedAggregationMethod
        
        aggregator = FederatedAggregator(FederatedAggregationMethod.FED_AVG)
        
        # Add client updates
        for i in range(3):
            update = ClientUpdate(
                client_id=f"client_{i}",
                model_weights={"layer": [float(i)]},
                weight_deltas={"layer": [0.1]},
                num_samples_trained=100,
                training_steps=10,
                loss=1.0 - (i * 0.1)
            )
            aggregator.add_client_update(update)
        
        result = aggregator.aggregate()
        assert "avg_loss" in result
        assert "num_clients" in result
    
    def test_differential_privacy_engine(self):
        from aiprod_pipelines.inference.distributed_lora import DifferentialPrivacyEngine, DifferentialPrivacyLevel
        
        engine = DifferentialPrivacyEngine(DifferentialPrivacyLevel.MEDIUM)
        
        # Test gradient clipping
        gradient = [1.0, 2.0, 3.0]
        clipped = engine.clip_gradient(gradient, clip_norm=1.0)
        assert clipped is not None
        
        # Test Laplace noise
        noisy_value = engine.add_laplace_noise(1.0, sensitivity=1.0)
        assert isinstance(noisy_value, float)
    
    def test_federated_trainer(self):
        from aiprod_pipelines.inference.distributed_lora import FederatedTrainer, FederatedAggregationMethod, DifferentialPrivacyLevel
        
        trainer = FederatedTrainer(
            num_clients=5,
            aggregation_method=FederatedAggregationMethod.FED_AVG,
            privacy_level=DifferentialPrivacyLevel.NONE
        )
        
        # Start round
        selected = trainer.start_round(sample_fraction=1.0)
        assert len(selected) > 0
        
        # Check convergence (after first round, still training)
        status = trainer.get_convergence_status()
        assert not status["converged"]


class TestLoRAMerging:
    """Tests for LoRA adapter merging"""
    
    def test_adapter_weight(self):
        from aiprod_pipelines.inference.distributed_lora import AdapterWeight
        
        weight = AdapterWeight(
            adapter_id="adapter_1",
            weight=2.0,
            scaling_factor=0.5
        )
        
        assert weight.adapter_id == "adapter_1"
        normalized = weight.normalized_weight(total_weight=2.0)
        assert 0 < normalized <= 1.0
    
    def test_composition_plan(self):
        from aiprod_pipelines.inference.distributed_lora import CompositionPlan, AdapterWeight, MergeStrategy
        
        adapters = [
            AdapterWeight("adapter_1", 1.0),
            AdapterWeight("adapter_2", 1.0)
        ]
        
        plan = CompositionPlan(
            plan_id="plan_001",
            adapters=adapters,
            strategy=MergeStrategy.LINEAR,
            description="Test merge"
        )
        
        assert plan.total_weight == 2.0
    
    def test_lora_merge_engine_linear(self):
        from aiprod_pipelines.inference.distributed_lora import LoRAMergeEngine, MergeStrategy
        
        engine = LoRAMergeEngine()
        
        adapters = [
            ("adapter_1", {"layer1": [1.0, 2.0]}, 1.0),
            ("adapter_2", {"layer1": [3.0, 4.0]}, 1.0)
        ]
        
        merged = engine.merge_adapters(adapters, MergeStrategy.LINEAR)
        assert len(merged) > 0
    
    def test_lora_merge_engine_hierarchical(self):
        from aiprod_pipelines.inference.distributed_lora import LoRAMergeEngine, MergeStrategy
        
        engine = LoRAMergeEngine()
        
        adapters = [
            ("parent", {"layer": [1.0]}, 1.0),
            ("child1", {"layer": [1.5]}, 1.0),
            ("child2", {"layer": [2.0]}, 1.0)
        ]
        
        merged = engine.merge_adapters(adapters, MergeStrategy.HIERARCHICAL)
        assert merged is not None
    
    def test_model_inheritance(self):
        from aiprod_pipelines.inference.distributed_lora import ModelInheritance
        
        inheritance = ModelInheritance()
        
        # Register relationships
        inheritance.register_inheritance("child_1", "parent")
        inheritance.register_inheritance("child_2", "parent")
        inheritance.register_inheritance("grandchild", "child_1")
        
        # Test queries
        parent = inheritance.get_parent_model("child_1")
        assert parent == "parent"
        
        children = inheritance.get_child_models("parent")
        assert len(children) == 2
        
        descendants = inheritance.get_descendants("parent")
        assert len(descendants) >= 2
    
    def test_adapter_composer(self):
        from aiprod_pipelines.inference.distributed_lora import AdapterComposer, LoRAMergeEngine, MergeStrategy
        
        engine = LoRAMergeEngine()
        composer = AdapterComposer(engine)
        
        composition_id = composer.compose_for_task(
            task_id="task_1",
            adapter_ids=["adapter_a", "adapter_b"],
            weights=[0.7, 0.3],
            strategy=MergeStrategy.LINEAR
        )
        
        assert composition_id is not None


class TestLoRARegistry:
    """Tests for model registry"""
    
    def test_model_status_enum(self):
        from aiprod_pipelines.inference.distributed_lora import ModelStatus
        
        assert ModelStatus.TRAINING.value == "training"
        assert ModelStatus.READY.value == "ready"
    
    def test_access_level_enum(self):
        from aiprod_pipelines.inference.distributed_lora import AccessLevel
        
        assert AccessLevel.PRIVATE.value == "private"
        assert AccessLevel.PUBLIC.value == "public"
    
    def test_lora_registry(self):
        from aiprod_pipelines.inference.distributed_lora import LoRARegistry
        
        registry = LoRARegistry()
        
        model_info = {
            "tenant_id": "tenant_1",
            "name": "test_model",
            "base_model": "base",
            "tags": ["nlp", "fine-tune"]
        }
        
        assert registry.register_model("model_001", model_info)
        model = registry.get_model("model_001")
        assert model is not None
    
    def test_model_discovery(self):
        from aiprod_pipelines.inference.distributed_lora import LoRARegistry, ModelDiscovery
        
        registry = LoRARegistry()
        
        # Register sample models
        for i in range(3):
            registry.register_model(f"model_{i}", {
                "tenant_id": "tenant_1",
                "name": f"Model {i}",
                "base_model": "base",
                "tags": ["nlp", "text"]
            })
        
        discovery = ModelDiscovery(registry)
        
        # Discover similar
        similar = discovery.discover_similar_models("model_0", is_public=False)
        assert len(similar) >= 0
    
    def test_registry_stats(self):
        from aiprod_pipelines.inference.distributed_lora import LoRARegistry
        
        registry = LoRARegistry()
        
        registry.register_model("model_1", {
            "tenant_id": "t1",
            "name": "Test",
            "base_model": "base"
        })
        
        stats = registry.get_registry_stats()
        assert stats["total_models"] == 1


class TestUserModelManagement:
    """Tests for user model management"""
    
    def test_user_model_tier(self):
        from aiprod_pipelines.inference.distributed_lora import UserModelTier
        
        assert UserModelTier.FREE.value == (2, 10)
        assert UserModelTier.PRO.value == (20, 1000000)
    
    def test_user_model_quota(self):
        from aiprod_pipelines.inference.distributed_lora import UserModelQuota, UserModelTier
        
        quota = UserModelQuota(user_id="user_1", tier=UserModelTier.FREE)
        
        assert quota.max_models == 2
        assert quota.can_create_model()
        
        quota.models_created = 2
        assert not quota.can_create_model()
    
    def test_user_model(self):
        from aiprod_pipelines.inference.distributed_lora import UserModel
        
        model = UserModel(
            model_id="model_1",
            user_id="user_1",
            tenant_id="tenant_1",
            name="My Model",
            description="Test model",
            base_model="base",
            created_at=datetime.utcnow(),
            storage_mb=50.0
        )
        
        assert model.model_id == "model_1"
        storage = model.get_storage_info()
        assert storage["storage_mb"] == 50.0
    
    def test_user_model_manager(self):
        from aiprod_pipelines.inference.distributed_lora import UserModelManager
        
        manager = UserModelManager(user_id="user_1", tenant_id="tenant_1")
        
        model = manager.create_model(
            model_id="model_1",
            name="Test",
            base_model="base_model"
        )
        
        assert model is not None
        retrieved = manager.get_model("model_1")
        assert retrieved.model_id == "model_1"
    
    def test_model_deduplication(self):
        from aiprod_pipelines.inference.distributed_lora import ModelDeduplication, UserModel
        from datetime import datetime
        
        dedup = ModelDeduplication(similarity_threshold=0.95)
        
        # Create sample models
        models = {
            "model_1": UserModel(
                model_id="model_1", user_id="u1", tenant_id="t1",
                name="Model 1", description="", base_model="base",
                created_at=datetime.utcnow()
            ),
            "model_2": UserModel(
                model_id="model_2", user_id="u2", tenant_id="t1",
                name="Model 2", description="", base_model="base",
                created_at=datetime.utcnow()
            )
        }
        
        groups = dedup.recommend_deduplication(models)
        assert isinstance(groups, list)
    
    def test_tenant_model_manager(self):
        from aiprod_pipelines.inference.distributed_lora import TenantModelManager
        
        manager = TenantModelManager(tenant_id="tenant_1")
        
        model = manager.create_user_model(
            user_id="user_1",
            model_id="model_1",
            name="Test",
            base_model="base"
        )
        
        assert model is not None
        
        # List user models
        models = manager.list_user_models("user_1")
        assert len(models) == 1


class TestDistributedLoRATrainer:
    """Tests for distributed LoRA training"""
    
    def test_training_mode_enum(self):
        from aiprod_pipelines.inference.distributed_lora import TrainingMode
        
        assert TrainingMode.CENTRALIZED.value == "centralized"
        assert TrainingMode.FEDERATED.value == "federated"
    
    def test_training_metrics(self):
        from aiprod_pipelines.inference.distributed_lora import TrainingMetrics
        
        metrics = TrainingMetrics(
            step=100,
            epoch=1,
            training_loss=0.5,
            learning_rate=1e-4,
            throughput_tokens_per_sec=1000.0
        )
        
        assert metrics.step == 100
        assert metrics.training_loss == 0.5
    
    def test_distributed_lora_trainer(self):
        from aiprod_pipelines.inference.distributed_lora import DistributedLoRATrainer, TrainingMode, LoRATrainingConfig
        
        trainer = DistributedLoRATrainer(
            model_id="model_1",
            config={"learning_rate": 1e-4, "max_steps": 1000},
            training_mode=TrainingMode.CENTRALIZED
        )
        
        # Setup
        setup = trainer.setup_distributed()
        assert setup is not None
        
        # Train step
        batch = {"num_samples": 32}
        metrics = trainer.train_step(batch)
        assert metrics.step == 1
    
    def test_distributed_lora_trainer_evaluation(self):
        from aiprod_pipelines.inference.distributed_lora import DistributedLoRATrainer, TrainingMode
        
        trainer = DistributedLoRATrainer(
            model_id="model_1",
            config={"learning_rate": 1e-4},
            training_mode=TrainingMode.CENTRALIZED
        )
        
        # Evaluate
        eval_results = trainer.evaluate()
        assert "eval_loss" in eval_results
        assert "eval_accuracy" in eval_results
    
    def test_federated_lora_trainer(self):
        from aiprod_pipelines.inference.distributed_lora import FederatedLoRATrainer
        
        trainer = FederatedLoRATrainer(
            model_id="model_1",
            num_clients=5
        )
        
        trainer.setup_clients(num_local_epochs=3)
        assert len(trainer.client_trainers) == 5
        
        # Run round
        result = trainer.run_federated_round()
        assert result["round"] == 1
        assert "avg_loss" in result
    
    def test_lora_inference_optimizer(self):
        from aiprod_pipelines.inference.distributed_lora import LoRAInferenceOptimizer
        
        optimizer = LoRAInferenceOptimizer()
        
        # Test quantization
        weights = {"layer1": [1.0, 2.0, 3.0]}
        quantized = optimizer.quantize_lora(weights, bits=8)
        assert len(quantized) > 0
        
        # Test performance estimation
        perf = optimizer.estimate_inference_performance(model_size_mb=100.0, batch_size=1)
        assert "tokens_per_sec" in perf
        assert "latency_ms" in perf


class TestIntegration:
    """Integration tests for distributed LoRA"""
    
    def test_end_to_end_user_model_creation(self):
        from aiprod_pipelines.inference.distributed_lora import TenantModelManager, UserLoRAPreset, LoRARank, LoRATarget, DistributedLoRAConfigManager
        
        # Create tenant manager
        mgr = TenantModelManager("tenant_1")
        
        # Create config manager
        cfg_mgr = DistributedLoRAConfigManager()
        
        # Get preset
        preset = cfg_mgr.get_preset("balanced")
        assert preset is not None
        
        # Create user model
        model = mgr.create_user_model(
            user_id="user_1",
            model_id="custom_model_1",
            name="My First LoRA",
            base_model="base_model"
        )
        
        assert model is not None
        assert model.user_id == "user_1"
    
    def test_federated_training_pipeline(self):
        from aiprod_pipelines.inference.distributed_lora import FederatedLoRATrainer, FederatedAggregationMethod, DifferentialPrivacyLevel
        
        # Create federated trainer
        trainer = FederatedLoRATrainer(
            model_id="fed_model_1",
            num_clients=10
        )
        
        trainer.setup_clients(num_local_epochs=3)
        
        # Run multiple rounds
        for _ in range(3):
            result = trainer.run_federated_round()
            assert result["num_clients"] == 10
        
        # Check convergence
        status = trainer.get_convergence_status()
        assert "current_round" in status
    
    def test_model_merge_and_inheritance(self):
        from aiprod_pipelines.inference.distributed_lora import ModelInheritance, LoRAMergeEngine, MergeStrategy, AdapterComposer, CompositionPlan, AdapterWeight
        
        # Setup inheritance
        inheritance = ModelInheritance()
        inheritance.register_inheritance("child", "parent")
        inheritance.register_inheritance("grandchild", "child")
        
        # Setup merge engine
        engine = LoRAMergeEngine()
        composer = AdapterComposer(engine)
        
        # Create composition
        composition_id = composer.compose_for_task(
            task_id="combined",
            adapter_ids=["parent", "child"],
            weights=[0.5, 0.5],
            strategy=MergeStrategy.LINEAR
        )
        
        assert composition_id is not None
        
        # Check distances
        distance = inheritance.compute_distance("child", "grandchild")
        assert distance >= 0
