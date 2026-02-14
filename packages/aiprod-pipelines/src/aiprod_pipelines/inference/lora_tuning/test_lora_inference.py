"""Test LoRA inference system."""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
from aiprod_pipelines.inference.lora_tuning.lora_inference import (
    LoRAInference,
    LoRABatchInference,
    LoRAEnsemble,
    LoRAConfig as LoRAInferenceConfig,
)


class TestLoRAInference:
    """Test basic LoRA inference."""
    
    def create_simple_model(self):
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        )
    
    def test_inference_creation(self):
        """Test inference engine creation."""
        model = self.create_simple_model()
        inference = LoRAInference(model, device="cpu")
        
        assert inference.model == model
        assert inference.device == "cpu"
        assert inference.active_adapter is None
    
    def test_forward_without_adapter(self):
        """Test forward pass without adapter."""
        model = self.create_simple_model()
        inference = LoRAInference(model, device="cpu")
        
        X = torch.randn(2, 10)
        output = inference.forward(X)
        
        assert output.shape == (2, 1)
    
    def test_load_adapter(self, tmp_path):
        """Test loading LoRA adapter."""
        model = self.create_simple_model()
        inference = LoRAInference(model, device="cpu")
        
        # Create dummy adapter
        adapter_path = tmp_path / "adapter.pt"
        adapter_data = {
            "lora_params": {"layer1.lora_A": torch.randn(4, 10)},
            "config": {"rank": 4},
        }
        torch.save(adapter_data, str(adapter_path))
        
        inference.load_adapter("test_adapter", str(adapter_path))
        
        assert "test_adapter" in inference.loaded_adapters
    
    def test_set_active_adapter(self, tmp_path):
        """Test setting active adapter."""
        model = self.create_simple_model()
        inference = LoRAInference(model, device="cpu")
        
        # Load adapter first
        adapter_path = tmp_path / "adapter.pt"
        adapter_data = {
            "lora_params": {},
            "config": {},
        }
        torch.save(adapter_data, str(adapter_path))
        inference.load_adapter("adapter1", str(adapter_path))
        
        # Set active
        inference.set_active_adapter("adapter1")
        
        assert inference.active_adapter == "adapter1"
    
    def test_merge_adapter(self, tmp_path):
        """Test merging adapter."""
        model = self.create_simple_model()
        inference = LoRAInference(model, device="cpu")
        
        # Load and merge
        adapter_path = tmp_path / "adapter.pt"
        adapter_data = {
            "lora_params": {},
            "config": {},
        }
        torch.save(adapter_data, str(adapter_path))
        inference.load_adapter("adapter1", str(adapter_path), merge=False)
        inference.merge_adapter("adapter1")
        
        assert inference.loaded_adapters["adapter1"]["merged"] is True
    
    def test_unmerge_adapter(self, tmp_path):
        """Test unmerging adapter."""
        model = self.create_simple_model()
        inference = LoRAInference(model, device="cpu")
        
        adapter_path = tmp_path / "adapter.pt"
        adapter_data = {
            "lora_params": {},
            "config": {},
        }
        torch.save(adapter_data, str(adapter_path))
        
        # Load, merge, then unmerge
        inference.load_adapter("adapter1", str(adapter_path), merge=True)
        inference.unmerge_adapter("adapter1")
        
        assert inference.loaded_adapters["adapter1"]["merged"] is False
    
    def test_get_adapter_info(self, tmp_path):
        """Test getting adapter info."""
        model = self.create_simple_model()
        inference = LoRAInference(model, device="cpu")
        
        adapter_path = tmp_path / "adapter.pt"
        adapter_data = {
            "lora_params": {"w1": torch.randn(10, 10)},
            "config": {},
        }
        torch.save(adapter_data, str(adapter_path))
        
        inference.load_adapter("adapter1", str(adapter_path))
        info = inference.get_adapter_info()
        
        assert "adapter1" in info
        assert info["adapter1"]["loaded"] is True
        assert info["adapter1"]["parameters"] > 0
    
    def test_list_adapters(self, tmp_path):
        """Test listing adapters."""
        model = self.create_simple_model()
        inference = LoRAInference(model, device="cpu")
        
        adapter_path = tmp_path / "adapter.pt"
        adapter_data = {"lora_params": {}, "config": {}}
        torch.save(adapter_data, str(adapter_path))
        
        inference.load_adapter("adapter1", str(adapter_path))
        inference.load_adapter("adapter2", str(adapter_path))
        
        adapters = inference.list_adapters()
        
        assert "adapter1" in adapters
        assert "adapter2" in adapters
        assert len(adapters) == 2
    
    def test_unload_adapter(self, tmp_path):
        """Test unloading adapter."""
        model = self.create_simple_model()
        inference = LoRAInference(model, device="cpu")
        
        adapter_path = tmp_path / "adapter.pt"
        adapter_data = {"lora_params": {}, "config": {}}
        torch.save(adapter_data, str(adapter_path))
        
        inference.load_adapter("adapter1", str(adapter_path))
        assert "adapter1" in inference.loaded_adapters
        
        inference.unload_adapter("adapter1")
        assert "adapter1" not in inference.loaded_adapters


class TestLoRABatchInference:
    """Test batch inference."""
    
    def create_simple_model(self):
        """Create a simple model."""
        return nn.Linear(10, 5)
    
    def test_batch_forward_same_adapter(self):
        """Test batch forward with same adapter."""
        model = self.create_simple_model()
        inference = LoRABatchInference(model, device="cpu")
        
        X = torch.randn(4, 10)
        outputs = inference.forward_batch(X)
        
        assert "default" in outputs
        assert outputs["default"].shape == (4, 5)
    
    def test_batch_forward_different_adapters(self):
        """Test batch forward with different adapters per sample."""
        model = self.create_simple_model()
        inference = LoRABatchInference(model, device="cpu")
        
        X = torch.randn(4, 10)
        adapter_names = ["adapter1", "adapter2", "adapter1", "adapter2"]
        
        outputs = inference.forward_batch(X, adapter_names)
        
        assert len(outputs) >= 1


class TestLoRAEnsemble:
    """Test ensemble inference."""
    
    def create_simple_model(self):
        """Create a simple model."""
        return nn.Linear(10, 5)
    
    def test_ensemble_creation(self, tmp_path):
        """Test ensemble creation."""
        model = self.create_simple_model()
        
        # Create adapters
        adapters = {}
        for i in range(2):
            path = tmp_path / f"adapter{i}.pt"
            adapter_data = {"lora_params": {}, "config": {}}
            torch.save(adapter_data, str(path))
            adapters[f"adapter{i}"] = str(path)
        
        ensemble = LoRAEnsemble(model, adapters, device="cpu")
        
        assert len(ensemble.inference.list_adapters()) == 2
    
    def test_ensemble_forward_average(self, tmp_path):
        """Test ensemble with average combination."""
        model = self.create_simple_model()
        
        # Create adapters
        adapters = {}
        for i in range(2):
            path = tmp_path / f"adapter{i}.pt"
            adapter_data = {"lora_params": {}, "config": {}}
            torch.save(adapter_data, str(path))
            adapters[f"adapter{i}"] = str(path)
        
        ensemble = LoRAEnsemble(model, adapters, device="cpu")
        
        X = torch.randn(2, 10)
        output = ensemble.forward_ensemble(X, ensemble_method="average")
        
        assert output.shape == (2, 5)
    
    def test_ensemble_forward_weighted(self, tmp_path):
        """Test ensemble with weighted combination."""
        model = self.create_simple_model()
        
        adapters = {}
        adapter_names = []
        for i in range(2):
            path = tmp_path / f"adapter{i}.pt"
            adapter_data = {"lora_params": {}, "config": {}}
            torch.save(adapter_data, str(path))
            name = f"adapter{i}"
            adapters[name] = str(path)
            adapter_names.append(name)
        
        ensemble = LoRAEnsemble(model, adapters, device="cpu")
        
        X = torch.randn(2, 10)
        weights = {name: 0.5 for name in adapter_names}
        
        output = ensemble.forward_ensemble(
            X,
            ensemble_method="weighted",
            adapter_weights=weights,
        )
        
        assert output.shape == (2, 5)
    
    def test_ensemble_forward_max(self, tmp_path):
        """Test ensemble with max combination."""
        model = self.create_simple_model()
        
        adapters = {}
        for i in range(2):
            path = tmp_path / f"adapter{i}.pt"
            adapter_data = {"lora_params": {}, "config": {}}
            torch.save(adapter_data, str(path))
            adapters[f"adapter{i}"] = str(path)
        
        ensemble = LoRAEnsemble(model, adapters, device="cpu")
        
        X = torch.randn(2, 10)
        output = ensemble.forward_ensemble(X, ensemble_method="max")
        
        assert output.shape == (2, 5)
    
    def test_ensemble_multiple_methods(self, tmp_path):
        """Test ensemble with multiple combination methods."""
        model = self.create_simple_model()
        
        adapters = {}
        for i in range(2):
            path = tmp_path / f"adapter{i}.pt"
            adapter_data = {"lora_params": {}, "config": {}}
            torch.save(adapter_data, str(path))
            adapters[f"adapter{i}"] = str(path)
        
        ensemble = LoRAEnsemble(model, adapters, device="cpu")
        
        X = torch.randn(2, 10)
        
        # Test each method
        for method in ["average", "weighted", "max"]:
            output = ensemble.forward_ensemble(X, ensemble_method=method)
            assert output.shape == (2, 5)


class TestLoRAInferenceConfig:
    """Test LoRA inference configuration."""
    
    def test_config_creation(self):
        """Test config creation."""
        config = LoRAInferenceConfig()
        
        assert config.use_lora is True
        assert config.scale_lora_weights == 1.0
        assert config.active_adapter is None
    
    def test_config_to_dict(self):
        """Test config to dictionary conversion."""
        config = LoRAInferenceConfig()
        config.scale_lora_weights = 2.0
        
        config_dict = config.to_dict()
        
        assert config_dict["use_lora"] is True
        assert config_dict["scale_lora_weights"] == 2.0
    
    def test_config_from_dict(self):
        """Test config from dictionary."""
        config_dict = {
            "use_lora": False,
            "scale_lora_weights": 0.5,
            "active_adapter": "test",
        }
        
        config = LoRAInferenceConfig.from_dict(config_dict)
        
        assert config.use_lora is False
        assert config.scale_lora_weights == 0.5
        assert config.active_adapter == "test"
    
    def test_config_roundtrip(self):
        """Test config serialization roundtrip."""
        original = LoRAInferenceConfig()
        original.use_lora = False
        original.scale_lora_weights = 2.0
        original.cache_adapters = False
        
        config_dict = original.to_dict()
        restored = LoRAInferenceConfig.from_dict(config_dict)
        
        assert restored.use_lora == original.use_lora
        assert restored.scale_lora_weights == original.scale_lora_weights
        assert restored.cache_adapters == original.cache_adapters


class TestLoRAInferenceIntegration:
    """Integration tests for inference system."""
    
    def test_adapter_switching(self, tmp_path):
        """Test switching between adapters."""
        model = nn.Linear(10, 5)
        inference = LoRAInference(model, device="cpu")
        
        # Load adapters
        for i in range(3):
            adapter_path = tmp_path / f"adapter{i}.pt"
            adapter_data = {"lora_params": {}, "config": {}}
            torch.save(adapter_data, str(adapter_path))
            inference.load_adapter(f"adapter{i}", str(adapter_path))
        
        # Switch adapters
        X = torch.randn(2, 10)
        
        for i in range(3):
            inference.set_active_adapter(f"adapter{i}")
            output = inference.forward(X)
            assert output.shape == (2, 5)
    
    def test_adapter_lifecycle(self, tmp_path):
        """Test full adapter lifecycle."""
        model = nn.Linear(10, 5)
        inference = LoRAInference(model, device="cpu")
        
        adapter_path = tmp_path / "adapter.pt"
        adapter_data = {
            "lora_params": {"w": torch.randn(5, 5)},
            "config": {},
        }
        torch.save(adapter_data, str(adapter_path))
        
        # Load
        inference.load_adapter("test", str(adapter_path))
        assert "test" in inference.list_adapters()
        
        # Set active
        inference.set_active_adapter("test")
        assert inference.active_adapter == "test"
        
        # Merge
        inference.merge_adapter("test")
        assert inference.loaded_adapters["test"]["merged"] is True
        
        # Info
        info = inference.get_adapter_info()
        assert info["test"]["merged"] is True
        
        # Unload
        inference.unload_adapter("test")
        assert "test" not in inference.list_adapters()
