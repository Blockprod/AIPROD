# Preset Graph Caching - Quick Reference

## Basic Usage (60 seconds to understand)

```python
from aiprod_pipelines.inference import preset_cached_t2v_one_stage

# Create preset (builds and caches graph)
graph1 = preset_cached_t2v_one_stage(encoder, model, scheduler, vae)

# Reuse with same models (returns cached graph!)
graph2 = preset_cached_t2v_one_stage(encoder, model, scheduler, vae)

# ✓ INSTANT: graph2 is same object as graph1 (50-500x faster)
assert graph1 is graph2
```

## Cache Management

```python
from aiprod_pipelines.inference import (
    preset_cache_config,    # Set max cache size
    preset_cache_clear,     # Clear all cached graphs
    preset_cache_size,      # Get current cache size
)

# Configure
preset_cache_config(64)        # Cache up to 64 graphs

# Check
print(preset_cache_size())     # How many graphs cached?

# Clear
preset_cache_clear()           # Free memory
```

## Available Cached Presets

| Function | Purpose |
|----------|---------|
| `preset_cached_t2v_one_stage()` | Text-to-video, single-pass |
| `preset_cached_t2v_two_stages()` | Text-to-video, two-pass upsampling |
| `preset_cached_distilled_fast()` | Ultra-fast 4-step inference |
| `preset_cached_ic_lora()` | Image/style control with LoRA |
| `preset_cached_keyframe_interpolation()` | Smooth interpolation between frames |
| `preset_cached_t2v_one_stage_adaptive()` | Dynamic guidance (better quality) |
| `preset_cached_t2v_two_stages_adaptive()` | Two-pass with adaptive guidance |
| `preset_cached_distilled_fast_adaptive()` | Fast with adaptive guidance |
| `preset_cached_ic_lora_adaptive()` | LoRA with adaptive guidance |
| `preset_cached_keyframe_interpolation_adaptive()` | Adaptive keyframe interpolation |
| `preset_cached_t2v_one_stage_quantized()` | Single-pass with INT8 (2-3x faster) |
| `preset_cached_t2v_two_stages_quantized()` | Two-pass with INT8 quantization |
| `preset_cached_distilled_fast_quantized()` | Fast with INT8 (5-8x total speedup) |
| `preset_cached_ic_lora_quantized()` | LoRA with quantization |
| `preset_cached_keyframe_interpolation_quantized()` | Keyframe interp with quantization |

## When Cache Hits (returns same graph)

✓ Same model instances  
✓ Same configuration parameters  
✓ Same preset type  

```python
encoder = load_encoder()
model = load_model()

# Cache hit (same encoder, model, scheduler, vae)
graph1 = preset_cached_t2v_one_stage(encoder, model, scheduler, vae)
graph2 = preset_cached_t2v_one_stage(encoder, model, scheduler, vae)
assert graph1 is graph2  # ✓ Same cached object
```

## When Cache Misses (creates new graph)

✗ Different model instances  
✗ Different configuration  
✗ Different preset type  

```python
encoder = load_encoder()
encoder2 = load_encoder()  # Different instance

# Cache miss (different encoders)
graph1 = preset_cached_t2v_one_stage(encoder, model, scheduler, vae)
graph2 = preset_cached_t2v_one_stage(encoder2, model, scheduler, vae)
assert graph1 is not graph2  # ✗ Different objects

# Cache miss (different config)
graph3 = preset_cached_t2v_one_stage(
    encoder, model, scheduler, vae,
    num_inference_steps=50  # Different from default
)
assert graph3 is not graph1  # ✗ Different config

# Cache miss (different preset type)
graph4 = preset_cached_distilled_fast(encoder, model, scheduler, vae)
assert graph4 is not graph1  # ✗ Different preset type
```

## Best Practices

### ✓ DO:
- **Load models once**: Reuse same model instances
- **Cache stays persistent**: Graphs stay cached across function calls
- **Different configs**: Different cache entries for different parameters
- **Clear when switching models**: Free memory when loading new models

### ✗ DON'T:
- **Don't reload models each time**: Defeats cache purpose (loads new instance)
- **Don't forget to clear cache**: Memory grows if not managed
- **Don't expect cache between processes**: Cache is in-process only

```python
# ✓ GOOD: Load once, use many times
encoder = load_text_encoder()
for i in range(100):
    graph = preset_cached_t2v_one_stage(encoder, model, scheduler, vae)
    # First: ~50ms (create), rest: ~0.1ms (cached) each

# ✗ BAD: Load in each call (no cache benefit)
for i in range(100):
    encoder = load_text_encoder()  # New instance each time!
    graph = preset_cached_t2v_one_stage(encoder, model, scheduler, vae)
    # All 100: ~50ms each (no cache, different encoder each time)
```

## Performance

| Operation | Time | Speedup |
|-----------|------|---------|
| Create preset (uncached) | ~50 ms | - |
| **Cache hit** | **~0.1 ms** | **500x** |
| Cache miss | ~50 ms | - |
| Clear cache | <1 ms | - |
| Cache lookup | ~0.1 ms | - |

## Typical Usage Pattern

```python
from aiprod_pipelines.inference import (
    preset_cached_t2v_one_stage,
    preset_cache_clear,
)

# Load models once
encoder = load_text_encoder()
model = load_denoising_model()
scheduler = load_scheduler()
vae = load_vae_decoder()

def generate_video(prompt):
    # Use cached preset (fast!)
    graph = preset_cached_t2v_one_stage(
        encoder, model, scheduler, vae,
        guidance_scale=7.5
    )
    return graph.run(prompt=prompt)

# All calls after first reuse cached graph
video1 = generate_video("A cat")         # ~50ms (create + run)
video2 = generate_video("A dog")         # ~0.1ms faster (cached)
video3 = generate_video("A bird")        # ~0.1ms faster (cached)

# When done or switching models
preset_cache_clear()  # Free memory
```

## Troubleshooting

### Problem: Cache not working (different graphs each time)
**Cause**: Loading models in function (different instance)  
**Fix**: Load models once before calling cached preset

```python
# ✗ Problem
def get_graph():
    encoder = load_text_encoder()  # New instance
    return preset_cached_t2v_one_stage(encoder, model, scheduler, vae)

graph1 = get_graph()
graph2 = get_graph()
# graph1 is not graph2 (cache miss due to different encoder)

# ✓ Solution
encoder = load_text_encoder()  # Load once
def get_graph():
    return preset_cached_t2v_one_stage(encoder, model, scheduler, vae)

graph1 = get_graph()
graph2 = get_graph()
# graph1 is graph2 (cache hit!)
```

### Problem: Memory keeps growing
**Cause**: Cache growing beyond configured size (or size too large)  
**Fix**: Lower cache limit or clear periodically

```python
from aiprod_pipelines.inference import (
    preset_cache_config,
    preset_cache_clear,
    preset_cache_size,
)

# Lower cache limit
preset_cache_config(16)  # Cache only 16 graphs

# Or clear periodically
if preset_cache_size() > 25:  # Out of 32
    preset_cache_clear()
    print("Cache cleared")
```

## API Reference

```python
# Import cache system
from aiprod_pipelines.inference import (
    PresetCache,                    # Cache class
    preset_cache_config,            # Set max size
    preset_cache_clear,             # Clear all graphs
    preset_cache_size,              # Get cache size
    preset_cached_t2v_one_stage,    # Cached preset (15 variants total)
    # ... and 14 more cached presets
)

# Create custom cache (advanced)
cache = PresetCache(max_size=64)

# Configure global cache
preset_cache_config(128)            # Cache up to 128 graphs

# Check cache
size = preset_cache_size()          # Returns current cache size

# Clear cache
preset_cache_clear()                # Clears all cached graphs
```

## More Information

- **Full Guide**: See [preset-caching.md](./preset-caching.md)
- **Implementation Details**: See [preset-caching-implementation.md](./preset-caching-implementation.md)
- **Code**: [presets.py](../src/aiprod_pipelines/inference/presets.py)
- **Tests**: [test_preset_cache.py](../tests/inference/caching/test_preset_cache.py)

---

**Summary**: Load models once → use cached presets multiple times → get 50-500x faster graph creation!
