#!/bin/bash
# =============================================================================
# AIPROD — AWS EC2 GPU Setup Script
# =============================================================================
# Run this ON the EC2 instance after SSH-ing in.
# Instance type: g5.xlarge (A10G 24GB VRAM, ~$1.01/h us-east-1)
# AMI: Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.5 (Ubuntu 22.04)
#
# Usage:
#   1. Launch g5.xlarge on AWS Console
#   2. SSH in: ssh -i your-key.pem ubuntu@<public-ip>
#   3. Upload this script: scp deploy/aws/setup_ec2_gpu.sh ubuntu@<ip>:~/
#   4. Run: bash setup_ec2_gpu.sh
# =============================================================================

set -euo pipefail

echo "============================================================"
echo "  AIPROD — EC2 GPU Instance Setup"
echo "============================================================"

# ── Verify GPU ────────────────────────────────────────────────────────────
echo ""
echo "[1/5] Verifying GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"

# ── Create workspace ─────────────────────────────────────────────────────
echo ""
echo "[2/5] Setting up workspace..."
mkdir -p ~/aiprod
cd ~/aiprod

# ── Install AIPROD packages ──────────────────────────────────────────────
echo ""
echo "[3/5] Installing AIPROD Python packages..."

# If code was uploaded via SCP/rsync
if [ -f "pyproject.toml" ]; then
    pip install -e packages/aiprod-core -e packages/aiprod-pipelines --quiet
else
    echo "⚠ No pyproject.toml found. Upload AIPROD code first:"
    echo "  rsync -avz --exclude='models/' --exclude='.venv*' --exclude='__pycache__' \\"
    echo "    /path/to/AIPROD/ ubuntu@<ip>:~/aiprod/"
    exit 1
fi

# Install extra deps if missing
pip install av einops safetensors transformers accelerate --quiet

# ── Verify models ────────────────────────────────────────────────────────
echo ""
echo "[4/5] Checking model weights..."

MODELS_OK=true

if [ ! -f "models/ltx2_research/ltx-2-19b-dev-fp8.safetensors" ]; then
    echo "❌ MISSING: models/ltx2_research/ltx-2-19b-dev-fp8.safetensors (25.2 GB)"
    echo "   Upload with: scp models/ltx2_research/ltx-2-19b-dev-fp8.safetensors ubuntu@<ip>:~/aiprod/models/ltx2_research/"
    MODELS_OK=false
fi

if [ ! -f "models/aiprod-sovereign/aiprod-text-encoder-v1/model.safetensors" ]; then
    echo "❌ MISSING: models/aiprod-sovereign/aiprod-text-encoder-v1/ (1.9 GB)"
    echo "   Upload with: scp -r models/aiprod-sovereign/aiprod-text-encoder-v1 ubuntu@<ip>:~/aiprod/models/aiprod-sovereign/"
    MODELS_OK=false
fi

if [ "$MODELS_OK" = true ]; then
    echo "✅ All required model weights found"
    ls -lh models/ltx2_research/*.safetensors
    ls -lh models/aiprod-sovereign/aiprod-text-encoder-v1/model.safetensors
fi

# ── Run validation ───────────────────────────────────────────────────────
echo ""
echo "[5/5] Running pipeline validation..."
python3 scripts/validate_pipeline_e2e.py --device cuda --height 64 --width 64 --frames 9 --steps 4

echo ""
echo "============================================================"
echo "  SETUP COMPLETE"
echo "============================================================"
echo ""
if [ "$MODELS_OK" = true ]; then
    echo "  Ready to generate! Run:"
    echo "    python3 scripts/generate_video_aws.py --prompt 'Your prompt here'"
    echo ""
else
    echo "  Upload missing models, then run:"
    echo "    python3 scripts/generate_video_aws.py --prompt 'Your prompt here'"
    echo ""
fi
