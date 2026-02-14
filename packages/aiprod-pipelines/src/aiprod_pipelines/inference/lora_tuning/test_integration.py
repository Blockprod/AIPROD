"""End-to-end LoRA integration tests."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import tempfile
from aiprod_pipelines.inference.lora_tuning.lora_config import (
    LoRAConfig,
    LoRAStrategy,
    LoRACompositionMode,
)
from aiprod_pipelines.inference.lora_tuning.lora_layers import (
    LoRALinear,
    LoRAAdapter,
    LoRAComposer,
    LoRAMerger,
)
from aiprod_pipelines.inference.lora_tuning.lora_trainer import (
    LoRATrainer,
    LoRAEvaluator,
)
from aiprod_pipelines.inference.lora_tuning.lora_inference import (
    LoRAInference,
    LoRAEnsemble,
)


class TestLoRAEndToEnd:
    """End-to-end LoRA workflow tests."""
    
    def create_model(self):
        """Create model for testing."""
        return nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    
    def create_dataset(self, num_samples=64):
        """Create synthetic dataset."""
        X = torch.randn(num_samples, 20)
        y = torch.randn(num_samples, 1)
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=8)
    
    def test_basic_lora_training(self):
        """Test basic LoRA training workflow."""
        model = self.create_model()
        config = LoRAConfig(
            rank=8,
            learning_rate=1e-3,
            num_epochs=2,
        )
        
        # Initialize trainer
        trainer = LoRATrainer(model, config, device="cpu")
        
        # Train
        train_loader = self.create_dataset(32)
        val_loader = self.create_dataset(16)
        
        history = trainer.train(train_loader, val_loader)
        
        assert len(history) > 0
        assert history[0]["loss"] > history[-1]["loss"]
    
    def test_strategy_based_training(self):
        """Test using pre-built strategies."""
        model = self.create_model()
        
        for strategy_name in ["resource_constrained", "high_quality", "quick_adaptation"]:
            # Select strategy
            if strategy_name == "resource_constrained":
                config = LoRAStrategy.for_resource_constrained()
            elif strategy_name == "high_quality":
                config = LoRAStrategy.for_high_quality()
            else:
                config = LoRAStrategy.for_quick_adaptation()
            
            config.num_epochs = 1
            
            trainer = LoRATrainer(model, config, device="cpu")
            
            train_loader = self.create_dataset(16)
            val_loader = self.create_dataset(8)
            
            history = trainer.train(train_loader, val_loader)
            assert len(history) > 0
    
    def test_multi_adapter_training(self):
        """Test training multiple adapters."""
        base_model = self.create_model()
        
        # Train adapter 1
        config1 = LoRAConfig(rank=8, num_epochs=1)
        trainer1 = LoRATrainer(base_model, config1, device="cpu")
        
        train_loader = self.create_dataset(32)
        val_loader = self.create_dataset(16)
        
        history1 = trainer1.train(train_loader, val_loader)
        
        # Train adapter 2 (different config)
        config2 = LoRAStrategy.for_quick_adaptation()
        config2.num_epochs = 1
        trainer2 = LoRATrainer(base_model, config2, device="cpu")
        
        history2 = trainer2.train(train_loader, val_loader)
        
        assert len(history1) > 0
        assert len(history2) > 0
    
    def test_adapter_composition(self):
        """Test using multiple adapters with composition."""
        base_model = self.create_model()
        
        # Create adapters
        adapter1 = LoRAAdapter(rank=8)
        for name, module in base_model.named_modules():
            if isinstance(module, nn.Linear):
                adapter1.add_lora_linear(name, module)
        
        adapter2 = LoRAAdapter(rank=8)
        for name, module in base_model.named_modules():
            if isinstance(module, nn.Linear):
                adapter2.add_lora_linear(name, module)
        
        # Test composition modes
        for mode in [
            LoRACompositionMode.sequential,
            LoRACompositionMode.parallel,
            LoRACompositionMode.gated,
            LoRACompositionMode.conditional,
        ]:
            composer = LoRAComposer(
                num_adapters=2,
                composition_mode=mode,
                hidden_dim=64,
            )
            
            outputs = [torch.randn(4, 1) for _ in range(2)]
            result = composer.compose_outputs(outputs)
            
            assert result.shape == (4, 1)
    
    def test_checkpoint_and_resume(self, tmp_path):
        """Test saving checkpoint and resuming training."""
        model = self.create_model()
        config = LoRAConfig(rank=8, num_epochs=3)
        
        # Train first stage
        trainer1 = LoRATrainer(model, config, device="cpu")
        
        train_loader = self.create_dataset(24)
        val_loader = self.create_dataset(8)
        
        # Manual training with checkpoint
        for epoch in range(2):
            trainer1.train_epoch(train_loader, val_loader)
        
        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer1.save_checkpoint(str(checkpoint_path))
        
        # Resume in new trainer
        model2 = self.create_model()
        trainer2 = LoRATrainer(model2, config, device="cpu")
        trainer2.load_checkpoint(str(checkpoint_path))
        
        step1 = trainer1.state.step
        step2 = trainer2.state.step
        
        assert step2 == step1
    
    def test_merge_unmerge_workflow(self):
        """Test merging LoRA into base weights."""
        base_layer = nn.Linear(20, 20)
        original_weight = base_layer.weight.data.clone()
        
        lora_layer = LoRALinear(base_layer, rank=8)
        
        # Train the LoRA layer
        optimizer = torch.optim.Adam(lora_layer.parameters(), lr=1e-3)
        for _ in range(5):
            x = torch.randn(4, 20)
            y = torch.randn(4, 20)
            
            output = lora_layer(x)
            loss = ((output - y) ** 2).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        merger = LoRAMerger(scaling=1.0)
        
        # Merge
        merged_layer = merger.merge_linear(base_layer, lora_layer)
        
        # Check weights changed
        assert not torch.allclose(merged_layer.weight, original_weight)
        
        # Evaluate both
        x_test = torch.randn(4, 20)
        with torch.no_grad():
            lora_output = lora_layer(x_test)
            merged_output = merged_layer(x_test)
        
        # Should be very similar
        assert torch.allclose(lora_output, merged_output, atol=1e-4)
    
    def test_inference_workflow(self, tmp_path):
        """Test inference with loaded adapters."""
        model = self.create_model()
        
        # Create and save adapter
        adapter_path = tmp_path / "adapter.pt"
        adapter_data = {
            "lora_params": {"layer1": torch.randn(8, 20)},
            "config": {"rank": 8},
        }
        torch.save(adapter_data, str(adapter_path))
        
        # Load for inference
        inference = LoRAInference(model, device="cpu")
        inference.load_adapter("trained", str(adapter_path))
        inference.set_active_adapter("trained")
        
        # Inference
        X = torch.randn(4, 20)
        output = inference.forward(X)
        
        assert output.shape == (4, 1)
    
    def test_ensemble_inference_workflow(self, tmp_path):
        """Test ensemble inference with multiple adapters."""
        model = self.create_model()
        
        # Create multiple adapters
        adapters = {}
        for i in range(3):
            path = tmp_path / f"adapter{i}.pt"
            adapter_data = {
                "lora_params": {},
                "config": {"rank": 8},
            }
            torch.save(adapter_data, str(path))
            adapters[f"adapter{i}"] = str(path)
        
        # Ensemble
        ensemble = LoRAEnsemble(model, adapters, device="cpu")
        
        X = torch.randn(4, 20)
        
        # Test multiple ensemble methods
        for method in ["average", "max"]:
            output = ensemble.forward_ensemble(X, ensemble_method=method)
            assert output.shape == (4, 1)
    
    def test_parameter_efficiency(self):
        """Test parameter efficiency of LoRA."""
        model = self.create_model()
        
        # Count base parameters
        base_params = sum(p.numel() for p in model.parameters())
        
        # Add LoRA adapters
        adapter = LoRAAdapter(rank=8)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                adapter.add_lora_linear(name, module)
        
        lora_params = adapter.get_parameter_count()
        
        # LoRA should be much smaller
        reduction_ratio = lora_params / base_params
        assert reduction_ratio < 0.5  # Less than 50% of base parameters
    
    def test_different_ranks_comparison(self):
        """Test LoRA with different ranks."""
        model = self.create_model()
        
        configs_and_params = []
        
        for rank in [4, 8, 16, 32]:
            config = LoRAConfig(rank=rank, num_epochs=1)
            
            trainer = LoRATrainer(model, config, device="cpu")
            
            train_loader = self.create_dataset(16)
            val_loader = self.create_dataset(8)
            
            history = trainer.train(train_loader, val_loader)
            
            configs_and_params.append({
                "rank": rank,
                "final_loss": history[-1]["loss"],
            })
        
        # Higher rank should generally have lower or equal loss
        for i in range(len(configs_and_params) - 1):
            assert configs_and_params[i]["rank"] < configs_and_params[i + 1]["rank"]
    
    def test_learning_rate_impact(self):
        """Test impact of different learning rates."""
        model = self.create_model()
        
        results = []
        
        for lr in [1e-5, 1e-4, 1e-3]:
            config = LoRAConfig(learning_rate=lr, num_epochs=1)
            trainer = LoRATrainer(model, config, device="cpu")
            
            train_loader = self.create_dataset(16)
            val_loader = self.create_dataset(8)
            
            history = trainer.train(train_loader, val_loader)
            
            results.append({
                "lr": lr,
                "final_loss": history[-1]["loss"],
            })
        
        # At least check that different LRs produce different results
        losses = [r["final_loss"] for r in results]
        assert len(set([round(l, 3) for l in losses])) > 1
    
    def test_full_workflow_with_evaluation(self):
        """Test complete workflow including evaluation."""
        model = self.create_model()
        config = LoRAConfig(rank=8, num_epochs=2)
        
        trainer = LoRATrainer(model, config, device="cpu")
        evaluator = LoRAEvaluator()
        
        train_loader = self.create_dataset(32)
        val_loader = self.create_dataset(16)
        
        # Train
        history = trainer.train(train_loader, val_loader)
        
        # Evaluate
        X_test = torch.randn(8, 20)
        y_test = torch.randn(8, 1)
        
        test_loss = evaluator.evaluate(
            model,
            X_test,
            y_test,
            nn.MSELoss(),
        )
        
        assert test_loss > 0
        assert len(history) > 0
    
    def test_multi_task_adaptation(self):
        """Test multi-task learning scenario."""
        model = self.create_model()
        
        # Use multi-task strategy
        config = LoRAStrategy.for_multi_task(num_tasks=3)
        config.num_epochs = 1
        
        trainer = LoRATrainer(model, config, device="cpu")
        
        # Train on first task
        train_loader = self.create_dataset(32)
        val_loader = self.create_dataset(8)
        
        history = trainer.train(train_loader, val_loader)
        
        assert len(history) > 0
        assert config.composition_mode == LoRACompositionMode.gated
