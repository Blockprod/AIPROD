"""
Latent distillation module - efficient inference via compression.

Compresses 4-8MB latent tensors to 1-2MB using learned quantization,
achieving 5-8x speedup with minimal quality loss.

Expected benefits:
- Compression ratio: 5-8x
- Quality retention: >95%
- Denoising speedup: 2-3x (from smaller tensor size)
- Pipeline speedup: 5-8x (with memory/cache benefits)
- Memory saved: 5-7MB per generation

Components:
  - LatentEncoder: Compress latents to quantized codes
  - LatentDecoder: Decompress codes back to latents
  - LatentDistillationEngine: High-level compression interface
  - LatentDistillationNode: GraphNode for pipeline integration
  - DistilledDenoiseNode: Specialized denoising on compressed domain

Usage:

    # Option 1: Insert compression nodes into pipeline
    from aiprod_pipelines.inference import preset
    
    graph = preset("t2v_one_stage")
    # Insert compression before denoise, decompression after
    
    # Option 2: Use distilled preset (all-in-one)
    # graph = preset("t2v_one_stage_distilled")
    
    result = graph.execute({
        "prompt": "A cat dancing",
        "negative_prompt": "",
    })
    
    # Result includes compression metrics
    print(f"Compressed {result['compression_metrics'].original_size_mb:.1f}MB "
          f"to {result['compression_metrics'].compressed_size_mb:.1f}MB")

Training:

    from aiprod_pipelines.inference.latent_distillation import (
        LatentDistillationEngine,
        LatentCompressionConfig,
    )
    
    config = LatentCompressionConfig(
        codebook_size=512,
        num_quantizers=4,
    )
    
    engine = LatentDistillationEngine(config)
    
    # Training loop
    optimizer = torch.optim.Adam(engine.encoder.parameters(), lr=1e-3)
    
    for latents_batch in dataset:
        codes, metrics = engine.encoder(latents_batch)
        loss = metrics["loss"]
        
        loss.backward()
        optimizer.step()
"""

from aiprod_pipelines.inference.latent_distillation import (
    LatentMetrics,
    LatentCompressionConfig,
    LatentEncoder,
    LatentDecoder,
    LatentDistillationEngine,
)
from aiprod_pipelines.inference.latent_distillation_node import (
    LatentDistillationNode,
    DistillationProfile,
    DistilledDenoiseNode,
)

__all__ = [
    # Core classes
    "LatentMetrics",
    "LatentCompressionConfig",
    "LatentEncoder",
    "LatentDecoder",
    "LatentDistillationEngine",
    # Node classes
    "LatentDistillationNode",
    "DistillationProfile",
    "DistilledDenoiseNode",
]
