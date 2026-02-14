# 🧪 Streaming Module Test Suite

Complete test coverage for AIPROD Streaming Architecture (Phase I, Innovation 1).

## 📋 Test Organization

```
tests/streaming/
├── conftest.py              # Shared fixtures and utilities
├── test_sources.py          # DataSource implementations
├── test_cache.py            # SmartLRUCache functionality
├── test_prefetcher.py       # AsyncPrefetcher behavior
├── test_adapter.py          # StreamingDatasetAdapter integration
├── test_performance.py      # Benchmarks and performance tests
├── run_tests.py             # Test runner script
└── __init__.py
```

## 🚀 Running Tests

### Setup

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-benchmark pytest-cov pytest-timeout zstandard

# From aiprod-trainer package root
cd packages/aiprod-trainer
```

### Run All Tests

```bash
# Basic run
pytest tests/streaming/

# Verbose output
pytest tests/streaming/ -v

# With coverage report
pytest tests/streaming/ --cov=aiprod_trainer.streaming --cov-report=html

# With timeout (prevent hanging)
pytest tests/streaming/ --timeout=300
```

### Run Specific Test Categories

```bash
# Unit tests only (no integration)
pytest tests/streaming/ -m "not integration"

# Integration tests only
pytest tests/streaming/ -m integration

# Performance benchmarks
pytest tests/streaming/test_performance.py --benchmark-only

# Specific test file
pytest tests/streaming/test_cache.py -v

# Specific test class
pytest tests/streaming/test_cache.py::TestCacheMetrics -v

# Specific test function
pytest tests/streaming/test_cache.py::test_cache_basic_put_get -v
```

### Run With Parallel Execution (faster)

```bash
# Requires: pip install pytest-xdist
pytest tests/streaming/ -n auto
```

## 📊 Test Coverage

### Total Tests: ~80 tests across 5 test files

| Module | Tests | Coverage |
|--------|-------|----------|
| **test_sources.py** | 15+ | LocalDataSource, compression, prefetch |
| **test_cache.py** | 20+ | LRU eviction, TTL, compression, metrics |
| **test_prefetcher.py** | 18+ | Prefetch queue, async worker, hit tracking |
| **test_adapter.py** | 18+ | Multi-source, DataLoader compat, caching |
| **test_performance.py** | 12+ | Benchmarks, throughput, latency, memory |

## 🎯 Test Categories

### Unit Tests (Fast)
- Basic functionality of each component
- Error handling and edge cases
- Compression/decompression correctness
- TTL expiration behavior
- LRU eviction policy

**Run:** `pytest tests/streaming/ -m "not slow"`  
**Typical time:** ~2-5 seconds

### Integration Tests (Medium)
- Multi-source data loading
- DataLoader compatibility
- Prefetch with cache interaction
- End-to-end adapter workflows

**Run:** `pytest tests/streaming/test_adapter.py`  
**Typical time:** ~10-20 seconds

### Performance Tests (Slower)
- Throughput benchmarks
- Latency characterization (P50/P99)
- Memory usage analysis
- Compression ratio measurement

**Run:** `pytest tests/streaming/test_performance.py`  
**Typical time:** ~30-60 seconds

## 📈 Expected Test Results

### Success Criteria

```
✅ test_local_source_fetch_uncompressed
   - Verify uncompressed data loading works

✅ test_cache_lru_eviction
   - Verify LRU eviction when cache full
   - Metrics show evictions > 0

✅ test_prefetcher_improves_hit_rate
   - Cache hit rate > 50% with prefetch

✅ test_adapter_dataloader_compatibility
   - Works with PyTorch DataLoader
   - Batch loading succeeds

✅ test_compression_ratio
   - Compression ratio > 2.0x for tensors

✅ test_throughput_with_cache
   - Throughput improvement vs no-cache
   - Hit rate increases after warmup
```

### Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| **Cache Hit Rate (after warmup)** | >85% | Second epoch performance |
| **Compression Ratio** | >2.3x | zstd on torch tensors |
| **P99 Latency (cache hit)** | <10ms | After in-memory cache |
| **Cache Miss Latency** | <100ms | From local disk |
| **Throughput (with prefetch)** | 1.5-2x vs baseline | Sequential access pattern |
| **Memory Overhead (cache)** | <10% | Relative to max_size_gb |

## 🐛 Debugging Failed Tests

### AsyncIO Issues

```bash
# Run with asyncio debug mode
PYTHONASYNCDEBUG=1 pytest tests/streaming/ -v

# Use pytest-asyncio with different event loop policy
pytest tests/streaming/ --asyncio-mode=auto
```

### Cache Issues

```python
# In test, print cache stats for debugging
stats = cache.get_cache_stats()
print(f"Hit rate: {stats['hit_rate']:.2%}")
print(f"Items: {stats['num_items']}")
print(f"Current size: {stats['current_size_gb']:.2f}GB")
```

### Prefetcher Hangs

```bash
# Use timeout to catch hanging tests
pytest tests/streaming/ --timeout=10 -v

# Check if worker task is running
# In test: assert not prefetcher._task.done()
```

### Memory Issues

```bash
# Run with memory profiler
pip install memory-profiler
pytest tests/streaming/ --memray

# Check for memory leaks
# Verify cache eviction is working
```

## 🔧 Customization

### Run Tests Against Different Cache Sizes

Edit `test_cache.py` or create custom config:

```python
@pytest.mark.parametrize("max_size_gb", [0.1, 1.0, 10.0])
async def test_cache_scaled(max_size_gb):
    cache = SmartLRUCache(max_size_gb=max_size_gb)
    # ... test code
```

### Add New Test Markers

In `pytest.ini`:

```ini
markers = 
    slow: slow tests
    gpu: requires GPU
    s3: requires S3 credentials
```

Then mark tests:

```python
@pytest.mark.s3
async def test_s3_source():
    # Test S3 functionality
```

## 📝 Test Fixtures

All shared fixtures are in `conftest.py`:

- `temp_data_dir`: Temporary directory with 100 test samples
- `sample_tensor_dict`: Pre-generated tensor dictionary
- `async_event_loop`: Asyncio event loop
- `mock_fetch_fn`: Mock async fetch function

### Using Fixtures

```python
@pytest.mark.asyncio
async def test_my_func(temp_data_dir, sample_tensor_dict):
    # temp_data_dir is Path with pre-populated data
    # sample_tensor_dict has pre-generated tensors
```

## 🚨 Common Issues and Solutions

### Issue: Tests are slow

**Solutions:**
- Use `-n auto` for parallel execution
- Skip performance tests during development: `-m "not slow"`
- Reduce temp_data_dir size in conftest.py

### Issue: Async tests fail with event loop errors

**Solution:**
```bash
pytest --asyncio-mode=auto
```

### Issue: Can't find test data

**Solution:**
```bash
# Ensure conftest.py is in tests/streaming/
# Or set PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### Issue: Compression tests fail

**Solution:**
```bash
# Ensure zstandard is installed
pip install zstandard

# Verify installation
python -c "import zstandard; print(zstandard.__version__)"
```

## 📊 Generating Reports

### Coverage Report (HTML)

```bash
pytest tests/streaming/ \
  --cov=aiprod_trainer.streaming \
  --cov-report=html \
  --cov-report=term

# Open report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### JUnit XML Report (for CI/CD)

```bash
pytest tests/streaming/ \
  --junit-xml=test-results.xml
```

### Benchmark Results

```bash
pytest tests/streaming/test_performance.py \
  --benchmark-only \
  --benchmark-autosave
```

Results saved in `.benchmarks/` for comparison.

## ✅ Pre-Commit Hook (Optional)

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
pytest tests/streaming/ -m "not slow" --timeout=30
if [ $? -ne 0 ]; then
  echo "Tests failed - commit blocked"
  exit 1
fi
```

Make executable:
```bash
chmod +x .git/hooks/pre-commit
```

## 🎓 Next Steps

1. **Run all tests**: `pytest tests/streaming/ -v`
2. **Check coverage**: `--cov=aiprod_trainer.streaming`
3. **Run benchmarks**: `pytest tests/streaming/test_performance.py --benchmark-only`
4. **Fix any failures**: See debugging section
5. **Commit with confidence**: All tests passing

## 📞 Support

For test-related issues:
1. Check test output for error messages
2. Run with `-vv` for more verbose output
3. Check conftest.py for fixture setup
4. Verify dependencies are installed
