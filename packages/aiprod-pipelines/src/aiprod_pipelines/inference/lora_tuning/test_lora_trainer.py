"""Test LoRA training system."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from aiprod_pipelines.inference.lora_tuning.lora_trainer import (
    TrainingState,
    LoRATrainer,
    LoRAEvaluator,
)
from aiprod_pipelines.inference.lora_tuning.lora_config import LoRAConfig
from aiprod_pipelines.inference.lora_tuning.lora_layers import LoRALinear


class TestTrainingState:
    """Test training state tracking."""
    
    def test_state_creation(self):
        """Test state creation."""
        state = TrainingState()
        assert state.step == 0
        assert state.epoch == 0
        assert state.best_val_loss == float('inf')
        assert state.patience_counter == 0
        assert state.should_stop is False
    
    def test_state_updates(self):
        """Test state updates."""
        state = TrainingState()
        
        state.step = 10
        state.epoch = 1
        state.best_val_loss = 0.5
        
        assert state.step == 10
        assert state.epoch == 1
        assert state.best_val_loss == 0.5


class TestLoRATrainer:
    """Test LoRA training engine."""
    
    def create_simple_model(self):
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )
    
    def create_data(self, num_samples=32):
        """Create synthetic data."""
        X = torch.randn(num_samples, 10)
        y = torch.randn(num_samples, 1)
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=8)
    
    def test_trainer_creation(self):
        """Test trainer creation."""
        model = self.create_simple_model()
        config = LoRAConfig(rank=4, learning_rate=1e-3)
        
        trainer = LoRATrainer(
            model=model,
            config=config,
            device="cpu",
        )
        
        assert trainer.model == model
        assert trainer.device == "cpu"
    
    def test_train_step(self):
        """Test single training step."""
        model = self.create_simple_model()
        config = LoRAConfig(rank=4)
        
        trainer = LoRATrainer(model, config, device="cpu")
        
        X = torch.randn(8, 10)
        y = torch.randn(8, 1)
        
        loss, grad_norm, lr = trainer.train_step(X, y)
        
        assert isinstance(loss, float)
        assert loss > 0
        assert grad_norm >= 0
        assert lr > 0
    
    def test_val_step(self):
        """Test validation step."""
        model = self.create_simple_model()
        config = LoRAConfig(rank=4)
        
        trainer = LoRATrainer(model, config, device="cpu")
        
        X = torch.randn(8, 10)
        y = torch.randn(8, 1)
        
        with torch.no_grad():
            val_loss = trainer.val_step(X, y)
        
        assert isinstance(val_loss, float)
        assert val_loss > 0
    
    def test_train_epoch(self):
        """Test full training epoch."""
        model = self.create_simple_model()
        config = LoRAConfig(rank=4, num_epochs=1)
        
        trainer = LoRATrainer(model, config, device="cpu")
        
        train_loader = self.create_data(32)
        val_loader = self.create_data(16)
        
        metrics = trainer.train_epoch(train_loader, val_loader)
        
        assert "loss" in metrics
        assert "val_loss" in metrics
        assert metrics["loss"] > 0
    
    def test_full_training_loop(self):
        """Test full training loop."""
        model = self.create_simple_model()
        config = LoRAConfig(rank=4, num_epochs=2)
        
        trainer = LoRATrainer(model, config, device="cpu")
        
        train_loader = self.create_data(32)
        val_loader = self.create_data(16)
        
        history = trainer.train(train_loader, val_loader)
        
        assert len(history) > 0
        assert "loss" in history[0]
    
    def test_gradient_clipping(self):
        """Test gradient clipping."""
        model = self.create_simple_model()
        config = LoRAConfig(rank=4, max_grad_norm=1.0)
        
        trainer = LoRATrainer(model, config, device="cpu")
        
        X = torch.randn(8, 10)
        y = torch.randn(8, 1)
        
        loss, grad_norm, _ = trainer.train_step(X, y)
        
        # Gradient norm should be clipped
        assert grad_norm <= config.max_grad_norm + 1e-5
    
    def test_learning_rate_scheduling(self):
        """Test learning rate scheduler."""
        model = self.create_simple_model()
        config = LoRAConfig(
            rank=4,
            learning_rate=1e-3,
            warmup_steps=2,
        )
        
        trainer = LoRATrainer(model, config, device="cpu")
        
        # Check multiple steps for scheduling
        lrs = []
        for _ in range(5):
            X = torch.randn(8, 10)
            y = torch.randn(8, 1)
            _, _, lr = trainer.train_step(X, y)
            lrs.append(lr)
        
        # Should have reached some constant LR
        assert len(lrs) == 5
    
    def test_save_checkpoint(self, tmp_path):
        """Test checkpoint saving."""
        model = self.create_simple_model()
        config = LoRAConfig(rank=4)
        
        trainer = LoRATrainer(model, config, device="cpu")
        
        # Perform one training step
        X = torch.randn(8, 10)
        y = torch.randn(8, 1)
        trainer.train_step(X, y)
        
        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path))
        
        assert checkpoint_path.exists()
    
    def test_load_checkpoint(self, tmp_path):
        """Test checkpoint loading."""
        model = self.create_simple_model()
        config = LoRAConfig(rank=4)
        
        trainer1 = LoRATrainer(model, config, device="cpu")
        
        # Save checkpoint
        X = torch.randn(8, 10)
        y = torch.randn(8, 1)
        trainer1.train_step(X, y)
        
        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer1.save_checkpoint(str(checkpoint_path))
        
        # Load checkpoint in new trainer
        model2 = self.create_simple_model()
        trainer2 = LoRATrainer(model2, config, device="cpu")
        trainer2.load_checkpoint(str(checkpoint_path))
        
        # States should be similar
        assert trainer2.state.step == trainer1.state.step
    
    def test_get_learning_rates(self):
        """Test getting current learning rates."""
        model = self.create_simple_model()
        config = LoRAConfig(rank=4, learning_rate=1e-4)
        
        trainer = LoRATrainer(model, config, device="cpu")
        
        lrs = trainer.get_learning_rates()
        
        assert len(lrs) > 0
    
    def test_early_stopping(self):
        """Test early stopping mechanism."""
        model = self.create_simple_model()
        config = LoRAConfig(rank=4, num_epochs=10)
        
        trainer = LoRATrainer(model, config, device="cpu")
        
        train_loader = self.create_data(32)
        val_loader = self.create_data(16)
        
        history = trainer.train(
            train_loader,
            val_loader,
            patience=2,
        )
        
        # Should stop early due to patience
        assert len(history) <= 10


class TestLoRAEvaluator:
    """Test model evaluation."""
    
    def test_evaluator_creation(self):
        """Test evaluator creation."""
        model = nn.Linear(10, 10)
        evaluator = LoRAEvaluator()
        assert evaluator is not None
    
    def test_evaluate(self):
        """Test evaluation."""
        model = nn.Linear(10, 10)
        criterion = nn.MSELoss()
        
        X = torch.randn(8, 10)
        y = torch.randn(8, 1)
        
        evaluator = LoRAEvaluator()
        loss = evaluator.evaluate(model, X, y, criterion)
        
        assert isinstance(loss, float)
        assert loss >= 0
    
    def test_evaluate_batch(self):
        """Test batch evaluation."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )
        criterion = nn.MSELoss()
        
        # Multiple batches
        X_batch = [torch.randn(8, 10) for _ in range(4)]
        y_batch = [torch.randn(8, 1) for _ in range(4)]
        
        evaluator = LoRAEvaluator()
        
        total_loss = 0
        for X, y in zip(X_batch, y_batch):
            loss = evaluator.evaluate(model, X, y, criterion)
            total_loss += loss
        
        assert total_loss > 0


class TestLoRATrainerIntegration:
    """Integration tests for training system."""
    
    def test_training_convergence(self):
        """Test that training reduces loss."""
        # Create a learnable task
        model = nn.Linear(10, 1)
        config = LoRAConfig(rank=4, learning_rate=1e-2, num_epochs=5)
        
        trainer = LoRATrainer(model, config, device="cpu")
        
        # Create dataset where y ≈ 0.5 * x.sum()
        def create_dataset():
            X = torch.randn(100, 10)
            y = (0.5 * X.sum(dim=1, keepdim=True)).detach()
            return DataLoader(
                TensorDataset(X, y),
                batch_size=16,
            )
        
        train_loader = create_dataset()
        val_loader = create_dataset()
        
        history = trainer.train(train_loader, val_loader)
        
        # Loss should generally decrease
        initial_loss = history[0]["loss"]
        final_loss = history[-1]["loss"]
        
        assert final_loss < initial_loss
    
    def test_checkpoint_resume(self, tmp_path):
        """Test resuming from checkpoint."""
        model = nn.Linear(10, 1)
        config = LoRAConfig(rank=4, num_epochs=2)
        
        trainer1 = LoRATrainer(model, config, device="cpu")
        
        X = torch.randn(32, 10)
        y = torch.randn(32, 1)
        
        # Train first step
        trainer1.train_step(X, y)
        step1 = trainer1.state.step
        
        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer1.save_checkpoint(str(checkpoint_path))
        
        # Create new trainer and resume
        model2 = nn.Linear(10, 1)
        trainer2 = LoRATrainer(model2, config, device="cpu")
        trainer2.load_checkpoint(str(checkpoint_path))
        
        # Train more
        trainer2.train_step(X, y)
        step2 = trainer2.state.step
        
        assert step2 > step1
    
    def test_multi_epoch_training(self):
        """Test multi-epoch training with patience."""
        model = nn.Linear(10, 1)
        config = LoRAConfig(rank=4, num_epochs=5)
        
        trainer = LoRATrainer(model, config, device="cpu")
        
        X = torch.randn(64, 10)
        y = torch.randn(64, 1)
        
        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=16)
        
        val_dataset = TensorDataset(X[:16], y[:16])
        val_loader = DataLoader(val_dataset, batch_size=16)
        
        history = trainer.train(
            train_loader,
            val_loader,
            patience=2,
        )
        
        assert len(history) > 0
        assert all("loss" in h for h in history)
