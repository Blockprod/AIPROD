"""LoRA training engine for fine-tuning.

Provides:
- Training loop implementation
- Gradient management
- Checkpoint saving/loading
- Metric tracking
- Validation
"""

from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Current training state."""
    
    step: int = 0
    epoch: int = 0
    best_val_loss: float = float("inf")
    patience_counter: int = 0
    should_stop: bool = False


class LoRATrainer:
    """Trainer for LoRA parameter-efficient fine-tuning."""
    
    def __init__(
        self,
        model: nn.Module,
        lora_params: List[torch.nn.Parameter],
        config: Dict[str, Any],
        device: str = "cuda",
    ):
        """
        Initialize LoRA trainer.
        
        Args:
            model: Base model to fine-tune
            lora_params: List of LoRA parameters to train
            config: Training configuration
            device: Device for training
        """
        self.model = model
        self.lora_params = lora_params
        self.config = config
        self.device = device
        
        # Set up optimizer
        self.optimizer = optim.AdamW(
            lora_params,
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 0.0),
        )
        
        # Learning rate scheduler
        self.scheduler = None
        if config.get("warmup_steps", 0) > 0:
            from torch.optim.lr_scheduler import LambdaLR
            
            def lr_lambda(step):
                if step < config["warmup_steps"]:
                    return float(step) / float(max(1, config["warmup_steps"]))
                return 1.0
            
            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        
        self.state = TrainingState()
        self.metrics_history = []
    
    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        criterion: Callable,
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: (inputs, targets) tuple
            criterion: Loss function
            
        Returns:
            Dictionary with step metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Forward pass - only LoRA params updated
        outputs = self.model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if self.config.get("max_grad_norm", 1.0) > 0:
            torch.nn.utils.clip_grad_norm_(
                self.lora_params,
                self.config["max_grad_norm"],
            )
        
        # Optimization step
        self.optimizer.step()
        
        if self.scheduler:
            self.scheduler.step()
        
        # Compute gradient norm for monitoring
        total_grad_norm = 0.0
        for param in self.lora_params:
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        self.state.step += 1
        
        return {
            "loss": loss.item(),
            "gradient_norm": total_grad_norm,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }
    
    def val_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        criterion: Callable,
    ) -> float:
        """
        Validation step.
        
        Args:
            batch: (inputs, targets) tuple
            criterion: Loss function
            
        Returns:
            Validation loss
        """
        self.model.eval()
        
        with torch.no_grad():
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
        
        return loss.item()
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[Callable] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            criterion: Loss function (default: MSE)
            
        Returns:
            Epoch metrics
        """
        if criterion is None:
            criterion = nn.MSELoss()
        
        epoch_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            metrics = self.train_step(batch, criterion)
            epoch_losses.append(metrics["loss"])
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"Epoch {self.state.epoch}, Batch {batch_idx + 1}: "
                    f"loss={metrics['loss']:.4f}, "
                    f"grad_norm={metrics['gradient_norm']:.4f}"
                )
        
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        
        # Validation
        val_loss = None
        if val_loader:
            val_losses = []
            for batch in val_loader:
                val_loss_batch = self.val_step(batch, criterion)
                val_losses.append(val_loss_batch)
            val_loss = sum(val_losses) / len(val_losses)
            
            logger.info(
                f"Epoch {self.state.epoch}: "
                f"train_loss={avg_train_loss:.4f}, "
                f"val_loss={val_loss:.4f}"
            )
        else:
            logger.info(f"Epoch {self.state.epoch}: train_loss={avg_train_loss:.4f}")
        
        self.state.epoch += 1
        
        return {
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None,
        criterion: Optional[Callable] = None,
        early_stopping_patience: int = 3,
    ) -> List[Dict[str, float]]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Number of epochs (default from config)
            criterion: Loss function
            early_stopping_patience: Epochs without improvement before stopping
            
        Returns:
            List of epoch metrics
        """
        num_epochs = num_epochs or self.config.get("num_epochs", 10)
        
        history = []
        
        for epoch in range(num_epochs):
            epoch_metrics = self.train_epoch(
                train_loader,
                val_loader,
                criterion,
            )
            history.append(epoch_metrics)
            
            # Early stopping
            if val_loader and epoch_metrics["val_loss"] is not None:
                if epoch_metrics["val_loss"] < self.state.best_val_loss:
                    self.state.best_val_loss = epoch_metrics["val_loss"]
                    self.state.patience_counter = 0
                else:
                    self.state.patience_counter += 1
                    if self.state.patience_counter >= early_stopping_patience:
                        logger.info(
                            f"Early stopping at epoch {epoch} "
                            f"(val_loss={epoch_metrics['val_loss']:.4f})"
                        )
                        break
        
        self.metrics_history = history
        return history
    
    def save_checkpoint(
        self,
        path: str,
        include_optimizer: bool = True,
    ) -> None:
        """
        Save training checkpoint.
        
        Args:
            path: Save path
            include_optimizer: Include optimizer state
        """
        checkpoint = {
            "step": self.state.step,
            "epoch": self.state.epoch,
            "lora_params": [p.data.clone() for p in self.lora_params],
            "config": self.config,
        }
        
        if include_optimizer:
            checkpoint["optimizer_state"] = self.optimizer.state_dict()
            if self.scheduler:
                checkpoint["scheduler_state"] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(
        self,
        path: str,
        load_optimizer: bool = True,
    ) -> None:
        """
        Load training checkpoint.
        
        Args:
            path: Checkpoint path
            load_optimizer: Load optimizer state
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.state.step = checkpoint.get("step", 0)
        self.state.epoch = checkpoint.get("epoch", 0)
        
        # Restore LoRA params
        for param, saved_data in zip(self.lora_params, checkpoint["lora_params"]):
            param.data.copy_(saved_data)
        
        if load_optimizer and "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            if self.scheduler and "scheduler_state" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        
        logger.info(f"Checkpoint loaded from {path}")
    
    def get_learning_rates(self) -> List[float]:
        """Get learning rate for each parameter group."""
        return [group["lr"] for group in self.optimizer.param_groups]


class LoRAEvaluator:
    """Evaluate LoRA-adapted model."""
    
    @staticmethod
    def evaluate(
        model: nn.Module,
        data_loader: DataLoader,
        criterion: Callable,
        device: str = "cuda",
    ) -> Dict[str, float]:
        """
        Evaluate model on dataset.
        
        Args:
            model: Model to evaluate
            data_loader: Data loader
            criterion: Loss function
            device: Device
            
        Returns:
            Evaluation metrics
        """
        model.eval()
        
        total_loss = 0.0
        total_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                total_batches += 1
        
        avg_loss = total_loss / total_batches
        
        return {
            "loss": avg_loss,
            "num_batches": total_batches,
        }
