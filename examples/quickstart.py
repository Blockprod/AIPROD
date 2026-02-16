#!/usr/bin/env python3
"""
AIPROD Quick Start - Text-to-Video Generation Example
Ready to run on local GPU (GTX 1070)

Usage:
    python examples/quickstart.py --prompt "describe your video" --seed 42
"""

import argparse
import sys
from pathlib import Path

# Add AIPROD packages to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages"))

try:
    import torch
    from aiprod_pipelines import TI2VidTwoStagesPipeline
    print(f"✅ PyTorch {torch.__version__} loaded")
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Make sure you're in the correct venv: . .venv_311\\Scripts\\Activate.ps1")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="AIPROD Text-to-Video Generation")
    
    parser.add_argument(
        "--prompt",
        type=str,
        default="A beautiful sunset over mountains with vibrant colors",
        help="Text prompt for video generation"
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="models/aiprod2",
        help="Path to AIPROD-2 checkpoint directory"
    )
    
    parser.add_argument(
        "--text-encoder-dir",
        type=str,
        default="models/aiprod-sovereign/aiprod-text-encoder-v1",
        help="Path to AIPROD text encoder directory"
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Video height (recommend 480 for GTX 1070)"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Video width (recommend 832 for GTX 1070)"
    )
    
    parser.add_argument(
        "--num-frames",
        type=int,
        default=121,
        help="Number of video frames"
    )
    
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.0,
        help="Guidance scale for prompt adherence"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="output.mp4",
        help="Output video path"
    )
    
    parser.add_argument(
        "--fp8",
        action="store_true",
        default=True,
        help="Use FP8 quantization for faster inference (default: True)"
    )
    
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["two_stages", "one_stage", "distilled"],
        default="two_stages",
        help="Pipeline to use (two_stages=quality, one_stage=speed, distilled=fastest)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("AIPROD - Text-to-Video Generation")
    print("="*60)
    print(f"Prompt:     {args.prompt}")
    print(f"Resolution: {args.height}x{args.width}")
    print(f"Frames:     {args.num_frames}")
    print(f"Pipeline:   {args.pipeline}")
    print(f"Output:     {args.output}")
    print("="*60 + "\n")
    
    # Check if checkpoint exists
    ckpt_path = Path(args.checkpoint_dir)
    if not ckpt_path.exists():
        print(f"❌ Checkpoint directory not found: {args.checkpoint_dir}")
        print(f"\nTo download models:")
        print(f"  1. Visit: https://huggingface.co/aiprod/aiprod-2-models")
        print(f"  2. Download checkpoint to: {args.checkpoint_dir}")
        print(f"  3. Run this script again")
        return
    
    # Initialize pipeline
    print("📦 Loading pipeline...")
    try:
        pipeline = TI2VidTwoStagesPipeline(
            ckpt_dir=args.checkpoint_dir,
            text_encoder_dir=args.text_encoder_dir,
            fp8_transformer=args.fp8
        )
    except Exception as e:
        print(f"❌ Failed to load pipeline: {e}")
        return
    
    print("✅ Pipeline loaded")
    
    # Generate video
    print("\n🎬 Generating video...")
    print("   (This may take 15-45 minutes on GTX 1070)")
    print("   Monitor GPU: nvidia-smi")
    
    try:
        video = pipeline(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            seed=args.seed
        )
        
        print("\n✅ Video generated successfully!")
        
        # Save video
        print(f"💾 Saving to {args.output}...")
        pipeline.save_video(video, args.output, fps=24)
        
        print(f"\n✅ COMPLETE!")
        print(f"   Output: {Path(args.output).absolute()}")
        
    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
