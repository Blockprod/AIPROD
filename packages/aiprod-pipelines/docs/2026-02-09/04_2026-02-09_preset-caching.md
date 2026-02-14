# Preset Graph Caching

## Overview

The preset caching system provides automatic caching of inference graphs to avoid redundant graph construction when the same preset is called multiple times with identical models.

**Key benefits:**
- Eliminates repeated graph construction overhead
- Reduces memory allocation for commonly-used presets
- Supports LRU (Least Recently Used) eviction policy
- Thread-safe cache management
- Zero performance impact when cache is unused

## Basic Usage

### Using Cached Presets

```python
from aiprod_pipelines.inference import (
    preset_cached_t2v_one_stage,
    preset_cached_t2v_two_stages,
    preset_cached_distilled_fast,
    preset_cache_clear,
    preset_cache_size,
)

# First call: Creates and caches graph
graph1 = preset_cached_t2v_one_stage(
    text_encoder=encoder,
    model=model,
    scheduler=scheduler,
    vae_decoder=vae,
    num_inference_steps=30,
)

# Second call: Returns cached graph (same models, same config)
graph2 = preset_cached_t2v_one_stage(
    text_encoder=encoder,
    model=model,
    scheduler=scheduler,
    vae_decoder=vae,
    num_inference_steps=30,
)

# graph1 and graph2 are the same object
assert graph1 is graph2

# Check cache status
print(f"Cached graphs: {preset_cache_size()}")  # Output: Cached graphs: 1

# Clear cache
preset_cache_clear()
```

### Available Cached Preset Functions

All standard preset functions have cached versions:

**Standard presets:**
- `preset_cached_t2v_one_stage()`
- `preset_cached_t2v_two_stages()` 
- `preset_cached_distilled_fast()`
- `preset_cached_ic_lora()`
- `preset_cached_keyframe_interpolation()`

**Adaptive presets:**
- `preset_cached_t2v_one_stage_adaptive()`
- `preset_cached_t2v_two_stages_adaptive()`
- `preset_cached_distilled_fast_adaptive()`
- `preset_cached_ic_lora_adaptive()`
- `preset_cached_keyframe_interpolation_adaptive()`

**Quantized presets:**
- `preset_cached_t2v_one_stage_quantized()`
- `preset_cached_t2v_two_stages_quantized()`
- `preset_cached_distilled_fast_quantized()`
- `preset_cached_ic_lora_quantized()`
- `preset_cached_keyframe_interpolation_quantized()`

## Cache Configuration

### Changing Cache Size

```python
from aiprod_pipelines.inference import preset_cache_config

# Configure cache for 64 graphs instead of default 32
preset_cache_config(max_size=64)
```

### Cache Management Functions

```python
from aiprod_pipelines.inference import (
    preset_cache_config,   # Set max cache size
    preset_cache_clear,    # Clear all cached graphs
    preset_cache_size,     # Get current cache size
)

# Configure cache
preset_cache_config(128)

# Get current size
current = preset_cache_size()  # Returns int: number of cached graphs

# Clear cache (e.g., to free memory)
preset_cache_clear()
```

## Cache Behavior

### How Caching Works

The cache uses three components to determine if a graph should be reused:

1. **Preset Type**: The preset name (e.g., "t2v_one_stage", "distilled_fast")
2. **Model Identity**: Python object identity of text_encoder, model, scheduler, vae_decoder, upsampler
3. **Configuration**: Hash of config dictionary (num_inference_steps, guidance_scale, etc.)

```python
from aiprod_pipelines.inference import preset_cached_t2v_one_stage

# Same models, same config → Cache HIT (returns cached graph)
graph1 = preset_cached_t2v_one_stage(encoder, model, scheduler, vae)
graph2 = preset_cached_t2v_one_stage(encoder, model, scheduler, vae)
assert graph1 is graph2  # Same object

# Different config → Cache MISS (creates new graph)
graph3 = preset_cached_t2v_one_stage(
    encoder, model, scheduler, vae,
    num_inference_steps=50  # Different than default
)
assert graph3 is not graph1  # Different object

# Different models → Cache MISS (creates new graph)
encoder2 = load_different_encoder()
graph4 = preset_cached_t2v_one_stage(encoder2, model, scheduler, vae)
assert graph4 is not graph1  # Different object
```

### Model Identity

The cache uses Python's built-in `id()` function to identify model objects. This means:
- **Same model object instance**: Cache HIT (reused)
- **Different model object instances**: Cache MISS (new graph created)

```python
# Load model once
encoder = load_text_encoder()

# All calls with this encoder instance → cache hits
graph1 = preset_cached_t2v_one_stage(encoder, model, scheduler, vae)
graph2 = preset_cached_t2v_one_stage(encoder, model, scheduler, vae)
graph3 = preset_cached_t2v_one_stage(encoder, model, scheduler, vae)
# All three are the same cached object

# Load different encoder instance
encoder2 = load_text_encoder()  # New instance
graph4 = preset_cached_t2v_one_stage(encoder2, model, scheduler, vae)
# This is a different cached object (different encoder identity)
```

### LRU Eviction Policy

When the cache reaches its maximum size, the least recently used (LRU) graph is evicted:

```python
from aiprod_pipelines.inference import (
    preset_cache_config,
    preset_cache_size,
    preset_cached_t2v_one_stage,
)

# Configure cache for 3 graphs maximum
preset_cache_config(3)

# Create 3 graphs (fill cache)
for i in range(3):
    encoder = load_encoder(i)
    graph = preset_cached_t2v_one_stage(encoder, model, scheduler, vae)

assert preset_cache_size() == 3

# Create 4th graph (oldest is evicted)
encoder4 = load_encoder(4)
graph = preset_cached_t2v_one_stage(encoder4, model, scheduler, vae)
assert preset_cache_size() == 3  # Still 3 (oldest was evicted)

# Accessing oldest graph again creates new instance
encoder0 = load_encoder(0)
graph = preset_cached_t2v_one_stage(encoder0, model, scheduler, vae)
# This is a newly created graph (old one was evicted)
```

## Performance Characteristics

### Memory Impact

- **Per cached graph**: ~1-5 MB (depends on node complexity)
- **Default cache size**: 32 graphs = ~32-160 MB
- **Cache overhead**: Minimal (hash table + LRU tracking)

### Speed Impact

- **Cache hit**: ~0.1 ms (hash lookup + return)
- **Cache miss**: Same as non-cached preset (~10-50 ms for graph construction)
- **Typical speedup**: 50-100x when reusing same preset with identical models

### Example Performance

```python
import time
from aiprod_pipelines.inference import (
    preset_cached_t2v_one_stage,
    preset_cache_clear,
)

encoder = load_text_encoder()
model = load_denoising_model()
scheduler = load_scheduler()
vae = load_vae_decoder()

# First call: Create + cache
start = time.time()
graph1 = preset_cached_t2v_one_stage(encoder, model, scheduler, vae)
first_time = time.time() - start
# ~50 ms (graph construction)

# Second call: Cached
start = time.time()
graph2 = preset_cached_t2v_one_stage(encoder, model, scheduler, vae)
cached_time = time.time() - start
# ~0.1 ms (hash lookup)

print(f"First call: {first_time*1000:.1f} ms")
print(f"Cached call: {cached_time*1000:.1f} ms")
print(f"Speedup: {first_time/cached_time:.0f}x")
# Output: Speedup: 500x
```

## Advanced Usage

### Direct Cache Access

For advanced use cases, you can interact with the cache directly:

```python
from aiprod_pipelines.inference import PresetCache
from aiprod_pipelines.inference import PresetFactory

# Create custom cache instance
cache = PresetCache(max_size=128)

# Use cache with factory methods
graph = cache.get_or_create(
    "t2v_one_stage",
    encoder, model, scheduler, vae,
    factory_fn=lambda: PresetFactory.t2v_one_stage(
        encoder, model, scheduler, vae, num_inference_steps=30
    ),
    num_inference_steps=30
)
```

### Multiple Cache Instances

For isolation or different cache strategies, create separate cache instances:

```python
from aiprod_pipelines.inference import PresetCache

# Cache for lightweight presets (small cache)
light_cache = PresetCache(max_size=16)

# Cache for heavy presets (large cache)
heavy_cache = PresetCache(max_size=64)

# Use different caches for different preset types
def get_light_preset(preset_type, *args, **kwargs):
    return light_cache.get_or_create(
        preset_type, *args,
        lambda: PresetFactory[preset_type](*args, **kwargs),
        **kwargs
    )
```

## Best Practices

### 1. Reuse Model Instances

For maximum cache effectiveness, load models once and reuse them:

```python
# Good: Load models once, reuse for multiple presets
encoder = load_text_encoder()
model = load_denoising_model()
scheduler = load_scheduler()
vae = load_vae_decoder()

# All these will cache hit
graph1 = preset_cached_t2v_one_stage(encoder, model, scheduler, vae)
graph2 = preset_cached_distilled_fast(encoder, model, scheduler, vae)
graph3 = preset_cached_t2v_one_stage(encoder, model, scheduler, vae)

# Bad: Loading models in each function call
def generate_video(prompt):
    encoder = load_text_encoder()  # New instance each time
    model = load_denoising_model()  # Cache misses
    scheduler = load_scheduler()
    vae = load_vae_decoder()
    graph = preset_cached_t2v_one_stage(encoder, model, scheduler, vae)
```

### 2. Clear Cache When Changing Models

Clear the cache when switching between different models to free memory:

```python
from aiprod_pipelines.inference import preset_cache_clear

# Using model A
graph1 = preset_cached_t2v_one_stage(encoder_a, model_a, scheduler, vae)

# Switching to model B
preset_cache_clear()  # Free model A's cached graphs
graph2 = preset_cached_t2v_one_stage(encoder_b, model_b, scheduler, vae)
```

### 3. Configure Cache Size Based on Hardware

Adjust cache size based on available memory:

```python
from aiprod_pipelines.inference import preset_cache_config

# Low VRAM: Small cache
preset_cache_config(8)

# Medium VRAM: Default cache
preset_cache_config(32)

# High VRAM: Large cache
preset_cache_config(128)
```

### 4. Monitor Cache Usage

In production, monitor cache hit rates:

```python
from aiprod_pipelines.inference import preset_cache_size

# Log current cache size periodically
cache_size = preset_cache_size()
print(f"Current cache size: {cache_size} graphs")

# Clear if approaching limit
if cache_size > 25:  # Out of 32
    preset_cache_clear()
    print("Cache cleared due to high usage")
```

## Testing with Cache

### Disabling Cache in Tests

For unit tests, you may want to disable caching to have pure factory-created graphs:

```python
from aiprod_pipelines.inference import PresetFactory, preset_cache_clear

def test_preset():
    # Use factory directly (no cache)
    graph = PresetFactory.t2v_one_stage(encoder, model, scheduler, vae)
    # ... test graph
    
    # Or clear cache between tests
    preset_cache_clear()
    graph = preset_cached_t2v_one_stage(encoder, model, scheduler, vae)
```

## Troubleshooting

### Cache Not Working?

**Symptom**: Same preset called twice creates different graph objects

**Cause**: Model identities differ (different instances of same model)

**Solution**: Ensure you're using the same model instance

```python
# Problem
def get_cached_preset():
    encoder = load_text_encoder()  # New instance each call
    return preset_cached_t2v_one_stage(encoder, model, scheduler, vae)

graph1 = get_cached_preset()
graph2 = get_cached_preset()
# graph1 is not graph2 (cache miss because encoder identities differ)

# Solution
encoder = load_text_encoder()  # Load once
def get_cached_preset():
    return preset_cached_t2v_one_stage(encoder, model, scheduler, vae)

graph1 = get_cached_preset()
graph2 = get_cached_preset()
# graph1 is graph2 (cache hit)
```

### Memory Growing Over Time?

**Symptom**: Memory usage increases with each new preset call

**Cause**: Cache is accumulating graphs (working as designed, or cache limit is too high)

**Solution**: Lower cache limit or clear cache periodically

```python
from aiprod_pipelines.inference import preset_cache_config, preset_cache_clear

# Lower cache limit
preset_cache_config(16)

# Or clear periodically
import time
last_clear = time.time()
def maybe_clear_cache():
    global last_clear
    if time.time() - last_clear > 3600:  # Clear every hour
        preset_cache_clear()
        last_clear = time.time()
```

## API Reference

### PresetCache Class

```python
from aiprod_pipelines.inference import PresetCache

cache = PresetCache(max_size: int = 32)
```

**Methods:**
- `get_or_create(preset_type, text_encoder, model, scheduler, vae_decoder, factory_fn, upsampler=None, **config)`
  - Returns cached graph or creates and caches new one
  - Returns: InferenceGraph

- `clear()`
  - Clears all cached graphs
  - Returns: None

- `size() -> int`
  - Returns number of cached graphs
  - Returns: int

### Cache Management Functions

```python
from aiprod_pipelines.inference import (
    preset_cache_config,      # (max_size: int) -> None
    preset_cache_clear,       # () -> None
    preset_cache_size,        # () -> int
)
```

### Cached Preset Functions

All preset functions have cached equivalents with `preset_cached_` prefix:

```python
preset_cached_t2v_one_stage(
    text_encoder,
    model,
    scheduler,
    vae_decoder,
    **config_overrides
) -> InferenceGraph

# And 14 more cached preset functions...
```

## See Also

- [Preset Inference Graphs](./presets.md)
- [Inference Graph Guide](./graph.md)
- [Performance Tuning](./performance.md)
