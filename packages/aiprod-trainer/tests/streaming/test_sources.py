"""
Unit tests for DataSource implementations.

Tests:
- LocalDataSource file loading and listing
- Compression support (zstd)
- Async fetch behavior
- Prefetch concurrency
"""

import asyncio
from pathlib import Path

import pytest
import torch
import zstandard as zstd

from aiprod_trainer.streaming.sources import DataSourceConfig, LocalDataSource


@pytest.mark.asyncio
async def test_local_source_fetch_uncompressed(temp_data_dir):
    """Test fetching uncompressed data from local source."""
    config = DataSourceConfig(
        name="test_local",
        source_type="local",
        path_or_uri=str(temp_data_dir / "latents"),
    )
    
    source = LocalDataSource(config)
    
    # Fetch first item
    data = await source.fetch_file("latent_0000.pt", decompress=True)
    
    assert "latents" in data
    assert "num_frames" in data
    assert data["num_frames"] == 5
    assert data["latents"].shape == (128, 5, 32, 32)


@pytest.mark.asyncio
async def test_local_source_fetch_compressed(temp_data_dir):
    """Test fetching compressed data from local source."""
    latents_dir = temp_data_dir / "latents"
    
    config = DataSourceConfig(
        name="test_local_zstd",
        source_type="local",
        path_or_uri=str(latents_dir),
    )
    
    source = LocalDataSource(config)
    
    # Create compressed test file
    original_data = {
        "latents": torch.randn(128, 5, 32, 32),
        "num_frames": 5,
    }
    
    # Compress using zstd
    import io
    buffer = io.BytesIO()
    torch.save(original_data, buffer)
    
    cctx = zstd.ZstdCompressor(level=3)
    compressed = cctx.compress(buffer.getvalue())
    
    # Save compressed file
    with open(latents_dir / "latent_compressed.pt.zst", "wb") as f:
        f.write(compressed)
    
    # Fetch and decompress
    data = await source.fetch_file("latent_compressed.pt.zst", decompress=True)
    
    assert "latents" in data
    assert torch.allclose(data["latents"], original_data["latents"])


@pytest.mark.asyncio
async def test_local_source_list_files(temp_data_dir):
    """Test listing files from local source."""
    config = DataSourceConfig(
        name="test_local_list",
        source_type="local",
        path_or_uri=str(temp_data_dir / "latents"),
    )
    
    source = LocalDataSource(config)
    files = source.list_files()
    
    assert len(files) == 100
    assert all(f.endswith(".pt") for f in files)
    assert files[0] == "latent_0000.pt"


@pytest.mark.asyncio
async def test_local_source_concurrent_fetch(temp_data_dir):
    """Test concurrent fetching from local source."""
    config = DataSourceConfig(
        name="test_concurrent",
        source_type="local",
        path_or_uri=str(temp_data_dir / "latents"),
    )
    
    source = LocalDataSource(config)
    
    # Fetch multiple files concurrently
    files = ["latent_0000.pt", "latent_0001.pt", "latent_0002.pt"]
    results = await asyncio.gather(*[source.fetch_file(f) for f in files])
    
    assert len(results) == 3
    assert all("latents" in r for r in results)


@pytest.mark.asyncio
async def test_local_source_prefetch_files(temp_data_dir):
    """Test prefetch_files method."""
    config = DataSourceConfig(
        name="test_prefetch",
        source_type="local",
        path_or_uri=str(temp_data_dir / "latents"),
    )
    
    source = LocalDataSource(config)
    
    # Prefetch multiple files
    files_to_prefetch = ["latent_0000.pt", "latent_0001.pt", "latent_0002.pt"]
    await source.prefetch_files(files_to_prefetch)
    
    # All should be accessible now (in executor cache)
    for f in files_to_prefetch:
        data = await source.fetch_file(f)
        assert data is not None


@pytest.mark.asyncio
async def test_local_source_file_not_found(temp_data_dir):
    """Test error handling for missing files."""
    config = DataSourceConfig(
        name="test_notfound",
        source_type="local",
        path_or_uri=str(temp_data_dir / "latents"),
    )
    
    source = LocalDataSource(config)
    
    with pytest.raises(FileNotFoundError):
        await source.fetch_file("nonexistent_file.pt")


@pytest.mark.asyncio
async def test_local_source_invalid_path():
    """Test error handling for invalid data path."""
    config = DataSourceConfig(
        name="test_invalid",
        source_type="local",
        path_or_uri="/nonexistent/path/to/data",
    )
    
    with pytest.raises(FileNotFoundError):
        LocalDataSource(config)


@pytest.mark.asyncio
async def test_local_source_fetch_performance(temp_data_dir, benchmark):
    """Benchmark performance of fetch operation."""
    config = DataSourceConfig(
        name="test_perf",
        source_type="local",
        path_or_uri=str(temp_data_dir / "latents"),
    )
    
    source = LocalDataSource(config)
    
    async def run_fetch():
        return await source.fetch_file("latent_0050.pt")
    
    # Benchmark async operation
    def sync_benchmark():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(run_fetch())
        loop.close()
        return result
    
    benchmark(sync_benchmark)


class TestLocalSourceCompression:
    """Test suite for compression functionality."""
    
    @pytest.mark.asyncio
    async def test_compression_ratio(self, temp_data_dir):
        """Measure actual compression ratio achieved."""
        latents_dir = temp_data_dir / "latents"
        
        # Load uncompressed file
        original = torch.load(latents_dir / "latent_0000.pt")
        
        # Measure uncompressed size
        import io
        buffer_uncompressed = io.BytesIO()
        torch.save(original, buffer_uncompressed)
        uncompressed_size = len(buffer_uncompressed.getvalue())
        
        # Compress it
        cctx = zstd.ZstdCompressor(level=3)
        compressed = cctx.compress(buffer_uncompressed.getvalue())
        compressed_size = len(compressed)
        
        # Calculate ratio
        ratio = uncompressed_size / compressed_size
        
        # Typically should achieve 2.3x+ for torch tensors
        assert ratio > 2.0, f"Compression ratio too low: {ratio:.2f}x"
    
    @pytest.mark.asyncio
    async def test_compression_lossless(self, temp_data_dir):
        """Verify compression is lossless (data integrity)."""
        latents_dir = temp_data_dir / "latents"
        config = DataSourceConfig(
            name="test_lossless",
            source_type="local",
            path_or_uri=str(latents_dir),
        )
        
        # Create test file
        original_data = {
            "latents": torch.randn(64, 5, 32, 32),
            "precision_test": torch.tensor([1.23456789, 9.87654321]),
        }
        
        import io
        buffer = io.BytesIO()
        torch.save(original_data, buffer)
        
        cctx = zstd.ZstdCompressor(level=3)
        compressed = cctx.compress(buffer.getvalue())
        
        # Save and reload
        test_file = latents_dir / "lossless_test.pt.zst"
        with open(test_file, "wb") as f:
            f.write(compressed)
        
        source = LocalDataSource(config)
        restored = await source.fetch_file("lossless_test.pt.zst", decompress=True)
        
        # Verify exact match
        assert torch.allclose(original_data["latents"], restored["latents"], atol=1e-6)
        assert torch.allclose(
            original_data["precision_test"],
            restored["precision_test"],
            atol=1e-6
        )
