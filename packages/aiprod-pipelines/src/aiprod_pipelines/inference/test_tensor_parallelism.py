"""
Tensor Parallelism Tests

Comprehensive test suite for distributed training infrastructure,
sharding strategies, communication, checkpointing, and load balancing.
"""

import pytest
from typing import Dict, Any
import sys


class TestShardingStrategies:
    """Tests for sharding strategy selection and planning"""
    
    def test_sharding_config_validation(self):
        from aiprod_pipelines.inference.tensor_parallelism import ShardingConfig, ShardingStrategy
        
        # Valid configuration
        config = ShardingConfig(
            strategy=ShardingStrategy.HYBRID,
            world_size=8,
            tp_size=2,
            pp_size=2
        )
        assert config.dp_size == 2  # 8 / (2*2) = 2
        assert config.is_distributed
    
    def test_sharding_config_invalid(self):
        from aiprod_pipelines.inference.tensor_parallelism import ShardingConfig, ShardingStrategy
        
        # Invalid: tp * pp > world_size
        with pytest.raises(ValueError):
            ShardingConfig(
                strategy=ShardingStrategy.HYBRID,
                world_size=8,
                tp_size=4,
                pp_size=4
            )
    
    def test_device_mesh_creation(self):
        from aiprod_pipelines.inference.tensor_parallelism import DeviceMesh
        
        mesh = DeviceMesh(shape=(4, 8, 2))
        assert mesh.size == 64
        
        # Test coordinate conversion
        coords = (0, 3, 1)
        rank = mesh.get_rank_from_coords(coords)
        assert rank >= 0
        
        # Test reverse
        recovered_coords = mesh.get_coords_from_rank(rank)
        assert recovered_coords == coords
    
    def test_data_parallel_planner(self):
        from aiprod_pipelines.inference.tensor_parallelism import DataParallelPlanner, ShardingConfig, ShardingStrategy
        
        config = ShardingConfig(
            strategy=ShardingStrategy.DATA_PARALLEL,
            world_size=8,
            tp_size=1,
            pp_size=1
        )
        planner = DataParallelPlanner(config)
        
        plan = planner.plan_tensor_sharding((128, 4096, 4096), "replicate")
        assert plan["strategy"] == "data_parallel"
        assert plan["requires_allreduce"] == True
    
    def test_tensor_parallel_planner_row_wise(self):
        from aiprod_pipelines.inference.tensor_parallelism import TensorParallelPlanner, ShardingConfig, ShardingStrategy
        
        config = ShardingConfig(
            strategy=ShardingStrategy.TENSOR_PARALLEL,
            world_size=8,
            tp_size=4,
            pp_size=1
        )
        planner = TensorParallelPlanner(config)
        
        plan = planner.plan_tensor_sharding((4096, 4096), "row")
        assert plan["strategy"] == "tensor_parallel"
        assert plan["sharding_type"] == "row_wise"
        assert plan["num_shards"] == 4
    
    def test_tensor_parallel_planner_col_wise(self):
        from aiprod_pipelines.inference.tensor_parallelism import TensorParallelPlanner, ShardingConfig, ShardingStrategy
        
        config = ShardingConfig(
            strategy=ShardingStrategy.TENSOR_PARALLEL,
            world_size=8,
            tp_size=4,
            pp_size=1
        )
        planner = TensorParallelPlanner(config)
        
        plan = planner.plan_tensor_sharding((4096, 4096), "col")
        assert plan["strategy"] == "tensor_parallel"
        assert plan["sharding_type"] == "col_wise"
        assert plan["requires_allreduce"] == True
    
    def test_hybrid_parallel_planner(self):
        from aiprod_pipelines.inference.tensor_parallelism import HybridParallelPlanner, ShardingConfig, ShardingStrategy
        
        config = ShardingConfig(
            strategy=ShardingStrategy.HYBRID,
            world_size=16,
            tp_size=2,
            pp_size=2
        )
        planner = HybridParallelPlanner(config)
        
        plan = planner.plan_tensor_sharding((4096, 4096), "dp_tp")
        assert plan["strategy"] == "hybrid"
        assert plan["composition"] == "dp_tp"


class TestCommunication:
    """Tests for distributed communication operations"""
    
    def test_allreduce_operation(self):
        from aiprod_pipelines.inference.tensor_parallelism import AllReduceOperation, CommunicationConfig, CommunicationBackend, ReduceOp
        
        config = CommunicationConfig(
            backend=CommunicationBackend.NCCL,
            world_size=8,
            rank=0
        )
        op = AllReduceOperation(config)
        
        data = [1.0, 2.0, 3.0]
        result = op.execute(data, ReduceOp.SUM)
        assert result == data  # Single rank, no actual reduction
    
    def test_allgather_operation(self):
        from aiprod_pipelines.inference.tensor_parallelism import AllGatherOperation, CommunicationConfig, CommunicationBackend
        
        config = CommunicationConfig(
            backend=CommunicationBackend.NCCL,
            world_size=4,
            rank=1
        )
        op = AllGatherOperation(config)
        
        data = [5.0, 6.0]
        result = op.execute(data)
        assert len(result) == config.world_size
    
    def test_broadcast_operation(self):
        from aiprod_pipelines.inference.tensor_parallelism import BroadcastOperation, CommunicationConfig, CommunicationBackend
        
        config = CommunicationConfig(
            backend=CommunicationBackend.NCCL,
            world_size=4,
            rank=2
        )
        op = BroadcastOperation(config)
        
        data = [10.0, 20.0, 30.0]
        result = op.execute(data, src_rank=0)
        assert result == data
    
    def test_communication_manager(self):
        from aiprod_pipelines.inference.tensor_parallelism import CommunicationManager, CommunicationConfig, CommunicationBackend
        
        config = CommunicationConfig(
            backend=CommunicationBackend.GLOO,
            world_size=2,
            rank=0
        )
        manager = CommunicationManager(config)
        
        data = [1.0, 2.0]
        result = manager.allreduce(data)
        assert result is not None
        
        stats = manager.get_communication_stats()
        assert stats["total_operations"] > 0
    
    def test_overlapped_communication(self):
        from aiprod_pipelines.inference.tensor_parallelism import OverlappedCommunication, CommunicationManager, CommunicationConfig, CommunicationBackend, ReduceOp
        
        config = CommunicationConfig(
            backend=CommunicationBackend.NCCL,
            world_size=4,
            rank=0
        )
        manager = CommunicationManager(config)
        overlapped = OverlappedCommunication(manager)
        
        # Submit async operations
        op_id1 = overlapped.submit_async_allreduce([1.0, 2.0], ReduceOp.SUM)
        op_id2 = overlapped.submit_async_allreduce([3.0, 4.0], ReduceOp.AVG)
        
        assert op_id1 != op_id2
        
        # Wait for operations
        results = overlapped.wait_all()
        assert len(results) == 2


class TestDistributedConfig:
    """Tests for distributed training configuration"""
    
    def test_distributed_config_creation(self):
        from aiprod_pipelines.inference.tensor_parallelism import DistributedConfig, DistributedBackend
        
        config = DistributedConfig(
            backend=DistributedBackend.NCCL,
            world_size=8,
            rank=0,
            local_rank=0
        )
        assert config.is_distributed
        assert config.is_main_process
    
    def test_distributed_environment_from_env(self):
        from aiprod_pipelines.inference.tensor_parallelism import DistributedEnvironment
        import os
        
        # Set environment variables
        os.environ["WORLD_SIZE"] = "4"
        os.environ["RANK"] = "2"
        os.environ["LOCAL_RANK"] = "1"
        
        env = DistributedEnvironment.from_env_vars()
        assert env.config.world_size == 4
        assert env.config.rank == 2
    
    def test_distributed_initializer(self):
        from aiprod_pipelines.inference.tensor_parallelism import DistributedInitializer, DistributedEnvironment, DistributedConfig, DistributedBackend
        
        config = DistributedConfig(
            backend=DistributedBackend.GLOO,
            world_size=2,
            rank=0
        )
        env = DistributedEnvironment(config=config)
        init = DistributedInitializer(env)
        
        assert init.initialize() == True
        assert init.get_rank() == 0
        assert init.get_world_size() == 2
        assert init.is_main_process() == True
    
    def test_process_group_manager(self):
        from aiprod_pipelines.inference.tensor_parallelism import ProcessGroupManager, DistributedInitializer, DistributedEnvironment, DistributedConfig, DistributedBackend
        
        config = DistributedConfig(
            backend=DistributedBackend.GLOO,
            world_size=8,
            rank=0
        )
        env = DistributedEnvironment(config=config)
        init = DistributedInitializer(env)
        init.initialize()
        
        manager = ProcessGroupManager(init)
        
        # Create hierarchical groups
        groups = manager.create_hierarchical_groups(
            tensor_parallel_size=2,
            pipeline_parallel_size=2
        )
        assert "tensor_parallel" in groups
        assert "pipeline_parallel" in groups


class TestCheckpointing:
    """Tests for distributed checkpoint management"""
    
    def test_checkpoint_metadata(self):
        from aiprod_pipelines.inference.tensor_parallelism import CheckpointMetadata, CheckpointFormat
        import time
        
        metadata = CheckpointMetadata(
            checkpoint_id="ckp_001",
            timestamp=time.time(),
            step=1000,
            epoch=5,
            world_size=8,
            tp_size=2,
            pp_size=2,
            dp_size=2,
            model_hidden_dim=4096,
            num_layers=32,
            total_params=7_000_000_000,
            sharding_strategy="hybrid",
            format=CheckpointFormat.DISTRIBUTED
        )
        
        assert metadata.checkpoint_id == "ckp_001"
        data_dict = metadata.to_dict()
        assert isinstance(data_dict, dict)
    
    def test_checkpoint_manager(self, tmp_path):
        from aiprod_pipelines.inference.tensor_parallelism import CheckpointManager, CheckpointMetadata, OptimizationState, CheckpointFormat
        import time
        
        manager = CheckpointManager(str(tmp_path), rank=0, world_size=1)
        
        metadata = CheckpointMetadata(
            checkpoint_id="test_001",
            timestamp=time.time(),
            step=100,
            epoch=1,
            world_size=1,
            tp_size=1,
            pp_size=1,
            dp_size=1,
            model_hidden_dim=512,
            num_layers=12,
            total_params=1_000_000,
            sharding_strategy="data_parallel"
        )
        
        model_state = {"layer1": {"w": [1.0, 2.0]}}
        opt_state = OptimizationState(step=100)
        
        path = manager.save_checkpoint(model_state, opt_state, metadata)
        assert path is not None
    
    def test_incremental_checkpointing(self, tmp_path):
        from aiprod_pipelines.inference.tensor_parallelism import CheckpointManager, IncrementalCheckpointing, CheckpointMetadata, OptimizationState
        import time
        
        manager = CheckpointManager(str(tmp_path), rank=0, world_size=1)
        inc_ckp = IncrementalCheckpointing(manager)
        
        metadata = CheckpointMetadata(
            checkpoint_id="inc_test",
            timestamp=time.time(),
            step=50,
            epoch=1,
            world_size=1,
            tp_size=1,
            pp_size=1,
            dp_size=1,
            model_hidden_dim=256,
            num_layers=8,
            total_params=500_000,
            sharding_strategy="data_parallel"
        )
        
        model_state = {"layer": [1.0, 2.0, 3.0]}
        opt_state = OptimizationState(step=50)
        
        path, ratio = inc_ckp.save_incremental(model_state, opt_state, metadata)
        assert 0.0 <= ratio <= 1.0


class TestLoadBalancing:
    """Tests for dynamic load balancing"""
    
    def test_device_workload(self):
        from aiprod_pipelines.inference.tensor_parallelism import DeviceWorkload
        
        workload = DeviceWorkload(device_id=0, rank=0)
        assert workload.is_idle
        
        workload.compute_utilization = 0.95
        assert workload.is_overloaded
        
        workload.compute_utilization = 0.5
        assert not workload.is_idle and not workload.is_overloaded
    
    def test_load_balancer(self):
        from aiprod_pipelines.inference.tensor_parallelism import LoadBalancer, LoadBalancingStrategy
        
        balancer = LoadBalancer(num_devices=8)
        
        # Update metrics for each device
        for i in range(8):
            util = 0.3 + (i * 0.05)  # Varying utilization
            balancer.update_device_metrics(i, util, 0.5, i % 3)
        
        metrics = balancer.get_load_metrics()
        assert metrics.total_queue_depth == 0 + 1 + 2 + 0 + 1 + 2 + 0 + 1
    
    def test_adaptive_load_balancer(self):
        from aiprod_pipelines.inference.tensor_parallelism import AdaptiveLoadBalancer
        
        balancer = AdaptiveLoadBalancer(num_devices=4)
        
        # Record metrics over multiple rounds
        for round in range(3):
            for i in range(4):
                balancer.update_device_metrics(i, 0.2 + (round * 0.1), 0.4, 0)
            metrics = balancer.get_load_metrics()
            balancer.record_strategy_performance(
                __import__('aiprod_pipelines.inference.tensor_parallelism', fromlist=['LoadBalancingStrategy']).LoadBalancingStrategy.GREEDY,
                metrics
            )
        
        strategy = balancer.select_best_strategy()
        assert strategy is not None
    
    def test_optimal_batch_size(self):
        from aiprod_pipelines.inference.tensor_parallelism import LoadBalancer
        
        balancer = LoadBalancer(num_devices=8)
        
        # Simulate high utilization
        for i in range(8):
            balancer.update_device_metrics(i, 0.95, 0.8, 5)
        
        global_bs, micro_bs = balancer.get_optimal_batch_size()
        assert global_bs > 0
        assert micro_bs > 0


class TestGradientAccumulation:
    """Tests for distributed gradient operations"""
    
    def test_gradient_buffer(self):
        from aiprod_pipelines.inference.tensor_parallelism import GradientBuffer
        
        buffer = GradientBuffer()
        
        buffer.add_gradient("layer1", [1.0, 2.0])
        buffer.add_gradient("layer1", [3.0, 4.0])
        
        assert buffer.accumulation_step == 2
        grad = buffer.get_accumulated_gradient("layer1")
        assert grad is not None
    
    def test_gradient_accumulator(self):
        from aiprod_pipelines.inference.tensor_parallelism import GradientAccumulator, GradientAccumulationConfig, GradientAccumulationMode
        
        config = GradientAccumulationConfig(
            mode=GradientAccumulationMode.DELAYED,
            num_accumulation_steps=4
        )
        acc = GradientAccumulator(config)
        
        # Accumulate gradients
        should_sync_list = []
        for i in range(8):
            should_sync = acc.accumulate_gradient("layer1", [float(i)])
            should_sync_list.append(should_sync)
        
        assert True in should_sync_list  # At least one sync
    
    def test_distributed_gradient_sync(self):
        from aiprod_pipelines.inference.tensor_parallelism import DistributedGradientSync
        
        sync = DistributedGradientSync(world_size=4, rank=0)
        
        gradients = {
            "layer1": [1.0, 2.0],
            "layer2": [3.0, 4.0]
        }
        
        result = sync.allreduce_gradients(gradients)
        assert len(result) == len(gradients)
        
        stats = sync.get_sync_stats()
        assert "num_allreduce_ops" in stats
    
    def test_gradient_compression(self):
        from aiprod_pipelines.inference.tensor_parallelism import GradientCompression, GradientCompressionConfig
        
        config = GradientCompressionConfig(
            enabled=True,
            compression_ratio=0.1
        )
        comp = GradientCompression(config)
        
        gradients = {"layer1": [1.0, 2.0], "layer2": [3.0, 4.0]}
        compressed = comp.compress_gradients(gradients)
        assert len(compressed) > 0
        
        stats = comp.get_compression_stats()
        assert "compression_ratio" in stats


class TestModelSharding:
    """Tests for model weight sharding"""
    
    def test_model_sharder(self):
        from aiprod_pipelines.inference.tensor_parallelism import ModelSharder, ModelShardingStrategy
        
        sharder = ModelSharder(
            strategy=ModelShardingStrategy.SHARD_ALL_LAYERS,
            tp_size=4
        )
        
        model_config = {
            "hidden_dim": 4096,
            "num_layers": 32,
            "attention_heads": 32,
            "vocab_size": 50000,
            "mlp_hidden_dim": 16384
        }
        
        plan = sharder.create_sharding_plan(model_config)
        assert len(plan) > 0  # Should have sharding specs
    
    def test_activation_checkpointing(self):
        from aiprod_pipelines.inference.tensor_parallelism import ActivationCheckpointing
        
        ckp = ActivationCheckpointing(num_checkpoints=2)
        
        checkpoint_layers = ckp.select_checkpointing_layers(32)
        assert len(checkpoint_layers) > 0
        
        memory_savings = ckp.get_memory_savings(100.0, 32)
        assert memory_savings > 0
    
    def test_layer_wise_partitioning(self):
        from aiprod_pipelines.inference.tensor_parallelism import LayerWisePartitioning
        
        partition = LayerWisePartitioning(num_devices=4, num_layers=32)
        
        # Each device gets 8 layers
        for device_id in range(4):
            start, end = partition.get_layers_for_device(device_id)
            assert end - start == 8
        
        # Test layer -> device mapping
        device = partition.get_device_for_layer(15)
        assert device == 1  # Layer 15 on device 1 (layers 8-16)


class TestIntegration:
    """Integration tests for tensor parallelism"""
    
    def test_end_to_end_sharding(self):
        from aiprod_pipelines.inference.tensor_parallelism import ShardingPlanner, ShardingConfig, ShardingStrategy
        
        config = ShardingConfig(
            strategy=ShardingStrategy.HYBRID,
            world_size=16,
            tp_size=2,
            pp_size=2
        )
        
        planner = ShardingPlanner(config)
        
        tensor_specs = {
            "embedding": "replicate",
            "linear1": "row",
            "linear2": "col"
        }
        
        plan = planner.plan_model_sharding(ShardingStrategy.HYBRID, tensor_specs)
        assert plan["strategy"] == "hybrid"
        assert plan["world_size"] == 16
    
    def test_multi_node_setup(self):
        from aiprod_pipelines.inference.tensor_parallelism import DistributedEnvironment, DistributedInitializer, ProcessGroupManager
        import os
        
        os.environ["WORLD_SIZE"] = "16"
        os.environ["RANK"] = "2"
        os.environ["LOCAL_RANK"] = "1"
        os.environ["NUM_NODES"] = "2"
        os.environ["NODE_RANK"] = "0"
        
        env = DistributedEnvironment.from_env_vars()
        init = DistributedInitializer(env)
        init.initialize()
        
        manager = ProcessGroupManager(init)
        groups = manager.create_hierarchical_groups(
            tensor_parallel_size=2,
            pipeline_parallel_size=2
        )
        
        assert "tensor_parallel" in groups
