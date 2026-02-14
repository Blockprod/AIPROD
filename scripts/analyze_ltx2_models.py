#!/usr/bin/env python3
"""
PHASE 0.2: LTX-2 Architecture Analysis Tool
============================================

Inspect downloaded LTX-2 models and extract architectural information
for AIPROD innovation planning.

Usage: python scripts/analyze_ltx2_models.py
"""

import os
import torch
from pathlib import Path
import json

def analyze_safetensors(file_path):
    """Analyze a safetensors model file"""
    print(f"\n{'='*70}")
    print(f"Analyzing: {Path(file_path).name}")
    print(f"{'='*70}")
    print(f"📁 Path: {file_path}")
    print(f"💾 Size: {os.path.getsize(file_path) / 1e9:.2f} GB")
    
    try:
        # Load safetensors file to inspect structure
        from safetensors import safe_open
        
        state_dict = {}
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        
        if isinstance(state_dict, dict):
            print(f"\n📊 State Dict Structure:")
            print(f"   Total parameters: {len(state_dict)}")
            print(f"\n   Top-level keys (first 20):")
            
            params_info = {}
            total_params = 0
            
            for i, (key, val) in enumerate(list(state_dict.items())[:20]):
                if isinstance(val, torch.Tensor):
                    num_params = val.numel()
                    total_params += num_params
                    shape = tuple(val.shape)
                    dtype = str(val.dtype)
                    params_info[key] = {
                        'shape': shape,
                        'dtype': dtype,
                        'params': num_params
                    }
                    print(f"   ├─ {key}")
                    print(f"   │  ├─ Shape: {shape}")
                    print(f"   │  ├─ Type: {dtype}")
                    print(f"   │  └─ Params: {num_params:,}")
            
            # Estimate architecture
            print(f"\n🔍 Architecture Clues:")
            
            # Count attention layers (common pattern)
            attention_layers = sum(1 for k in state_dict.keys() if 'attn' in k.lower() or 'self_attn' in k.lower())
            if attention_layers > 0:
                print(f"   • Attention layers detected: {attention_layers}")
            
            # Count transformer blocks
            block_indices = set()
            for key in state_dict.keys():
                if 'layer.' in key or 'block.' in key or 'blocks.' in key:
                    try:
                        # Extract block number
                        parts = key.split('.')
                        for i, part in enumerate(parts):
                            if part.isdigit():
                                block_indices.add(int(part))
                    except:
                        pass
            
            if block_indices:
                num_blocks = max(block_indices) + 1
                print(f"   • Transformer blocks: ~{num_blocks}")
            
            # Embedding dimension
            for key, val in state_dict.items():
                if 'embed' in key.lower() and len(val.shape) == 2:
                    print(f"   • Embedding dimension: {val.shape[-1]}")
                    break
            
            # Activation function clues
            activation_keys = [k for k in state_dict.keys() if any(x in k.lower() for x in ['activation', 'gelu', 'relu', 'mlp'])]
            if activation_keys:
                print(f"   • Potential activation layers: {len(activation_keys)}")
            
            print(f"\n📈 Total Parameters (visible): {total_params:,}")
            print(f"   ≈ {total_params / 1e9:.2f} B parameters")
        
        return state_dict
        
    except Exception as e:
        print(f"⚠️  Error analyzing: {e}")
        return None

def main():
    print("\n" + "="*70)
    print("PHASE 0.2: LTX-2 ARCHITECTURE ANALYSIS")
    print("="*70)
    
    models_dir = Path("models/ltx2_research")
    
    if not models_dir.exists():
        print(f"❌ Error: {models_dir} not found!")
        return 1
    
    safetensors_files = list(models_dir.glob("*.safetensors"))
    
    if not safetensors_files:
        print(f"❌ No .safetensors files found in {models_dir}")
        return 1
    
    print(f"\n🔎 Found {len(safetensors_files)} models to analyze:")
    for f in safetensors_files:
        print(f"   • {f.name}")
    
    # Analyze each model
    results = {}
    for model_file in safetensors_files:
        try:
            result = analyze_safetensors(str(model_file))
            results[model_file.name] = result is not None
        except Exception as e:
            print(f"❌ Error: {e}")
            results[model_file.name] = False
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    
    print(f"\n✅ Successfully analyzed: {sum(results.values())}/{len(results)}")
    
    print(f"""
🎯 NEXT STEPS FOR PHASE 0.2:

1. Review the architecture information above
2. Open: docs/PHASE_0_RESEARCH_STRATEGY.md
3. Fill in sections:
   ├─ Task 0.2.1: Backbone Architecture Study
   ├─ Task 0.2.2: Video VAE Analysis
   ├─ Task 0.2.3: Text Encoding Integration
   ├─ Task 0.2.4: Temporal Modeling
   └─ Task 0.2.5: Training Methodology

4. Document your findings and insights

5. Then proceed to Phase 0.3: Innovation Domains

Once analysis complete, report: "PHASE 0.2 COMPLETE"
    """)
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
