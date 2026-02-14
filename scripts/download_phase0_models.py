#!/usr/bin/env python3
"""
PHASE 0.1: Download LTX-2 Reference Models for AIPROD Research
================================================================

This script downloads the OPTION 1 recommended models:
- ltx-2-19b-dev-fp8.safetensors (18GB)
- ltx-2-spatial-upscaler-x2-1.0.safetensors (6GB)

Total: ~24GB
"""

import os
import sys
from huggingface_hub import snapshot_download

def main():
    print("=" * 70)
    print("PHASE 0.1: TÉLÉCHARGEMENT MODÈLES LTX-2 (OPTION 1)")
    print("=" * 70)
    print()
    
    repo_id = "Lightricks/LTX-2"
    local_dir = "models/ltx2_research"
    files = [
        "ltx-2-19b-dev-fp8.safetensors",
        "ltx-2-spatial-upscaler-x2-1.0.safetensors"
    ]
    
    print(f"📦 Repository: {repo_id}")
    print(f"📁 Local directory: {local_dir}")
    print(f"📄 Files to download:")
    for f in files:
        print(f"   ├─ {f}")
    print()
    print("⏳ Starting download (30-60 minutes, depends on internet speed)...")
    print()
    
    try:
        # Download models
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=local_dir,
            allow_patterns=files,
            resume_download=True,
            force_download=False
        )
        
        print()
        print("✅ TÉLÉCHARGEMENT TERMINÉ AVEC SUCCÈS!")
        print()
        print("Fichiers téléchargés dans", local_dir + ":")
        
        # List downloaded files
        total_size = 0
        for root, dirs, filelist in os.walk(local_dir):
            for f in sorted(filelist):
                if f.endswith('.safetensors'):
                    full_path = os.path.join(root, f)
                    size_bytes = os.path.getsize(full_path)
                    size_gb = size_bytes / (1024**3)
                    total_size += size_bytes
                    print(f"   ✓ {f}")
                    print(f"     └─ {size_gb:.2f} GB ({size_bytes:,} bytes)")
        
        print()
        print(f"📊 Total downloaded: {total_size / (1024**3):.2f} GB")
        print()
        print("✅ PHASE 0.1 COMPLETE - Ready for Phase 0.2 Analysis")
        print("=" * 70)
        return 0
        
    except Exception as e:
        print()
        print(f"❌ ERROR: {e}")
        print()
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())
