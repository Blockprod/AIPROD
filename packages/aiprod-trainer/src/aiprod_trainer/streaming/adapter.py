"""
StreamingDatasetAdapter: Unified interface for streaming from multiple sources.

Replaces PrecomputedDataset for high-scale data loading with:
- Multi-source support (Local, HF, S3, GCS)
- Intelligent LRU caching with zstd compression
- Async prefetching
- Performance metrics
"""

import asyncio
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from aiprod_trainer import logger
from aiprod_trainer.streaming.cache import AsyncPrefetcher, SmartLRUCache
from aiprod_trainer.streaming.sources import DataSource, DataSourceConfig, create_data_source


class StreamingDatasetAdapter(Dataset):
    """
    High-performance dataset adapter for streaming from multiple sources.
    
    Features:
    - Multi-source support (local, Hugging Face, S3, GCS)
    - Intelligent caching with zstd compression
    - Async prefetch ahead-of-time
    - Automatic memory management
    - Hit-rate monitoring
    
    Example usage (multiple sources):
    ```python
    sources = [
        DataSourceConfig('local_1',  'local', '/data/local_1'),
        DataSourceConfig('hf_1', 'huggingface', 'username/dataset'),
        DataSourceConfig('s3_1', 's3', 's3://bucket/prefix'),
    ]
    
    dataset = StreamingDatasetAdapter(
        sources=sources,
        data_mapping={'latents': 'latent_conditions', 'conditions': 'text_conditions'},
        cache_size_gb=100,
        prefetch_ahead=10,
    )
    
    # In training loop:
    batch = [dataset[i] for i in range(batch_size)]
    
    # Monitor:
    stats = dataset.get_cache_stats()
    print(f"Cache hit rate: {stats['hit_rate']:.2%}")
    ```
    """
    
    def __init__(
        self,
        sources: list[DataSourceConfig],
        data_mapping: dict[str, str] | None = None,
        cache_size_gb: float = 100.0,
        max_ttl_seconds: float = 3600.0,
        prefetch_ahead: int = 10,
        enable_prefetch: bool = True,
        compression_type: str = 'zstd',
    ):
        """
        Initialize StreamingDatasetAdapter.
        
        Args:
            sources: List of DataSourceConfig for each data source
            data_mapping: Dict mapping source names to output keys
                         Default: {'latents': 'latent_conditions', 'conditions': 'text_conditions'}
            cache_size_gb: Maximum cache size in GB
            max_ttl_seconds: Max time entry lives in cache
            prefetch_ahead: Number of items to prefetch ahead
            enable_prefetch: Whether to enable async prefetching
            compression_type: 'zstd' (recommended) or 'none'
        """
        super().__init__()
        
        self.sources = sources
        self.data_mapping = data_mapping or {
            'latents': 'latent_conditions',
            'conditions': 'text_conditions',
        }
        self.enable_prefetch = enable_prefetch
        
        # Initialize data sources
        self._data_sources: dict[str, DataSource] = {}
        self._init_sources()
        
        # Initialize cache
        self._cache = SmartLRUCache(
            max_size_gb=cache_size_gb,
            max_ttl_seconds=max_ttl_seconds,
            compression_type=compression_type,
            prefetch_ahead=prefetch_ahead,
        )
        
        # Initialize prefetcher
        self._prefetcher = None
        if enable_prefetch:
            self._prefetcher = AsyncPrefetcher(
                cache=self._cache,
                fetch_fn=self._fetch_item,
                queue_size=prefetch_ahead * 2,
            )
        
        # Discover dataset size
        self._total_samples = self._discover_dataset_size()
        
        # Track async loop
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return self._total_samples
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get item at index (blocking, but with prefetch from cache if available).
        
        Note: If using in DataLoader with num_workers > 0, this runs in subprocess.
        Prefetch works best with num_workers=0 and async training loop.
        """
        # Get event loop for this process
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Build cache key from index
        cache_key = f"item_{idx:08d}"
        
        # Try async get through prefetcher (if available and running)
        if self._prefetcher:
            try:
                data = loop.run_until_complete(self._prefetcher.get(cache_key))
            except Exception:
                # Fallback to direct fetch
                logger.debug(f"Prefetcher failed for idx={idx}, falling back to direct fetch")
                data = loop.run_until_complete(self._fetch_item(cache_key))
        else:
            # Direct fetch (no prefetch)
            data = loop.run_until_complete(self._fetch_item(cache_key))
        
        # Add index for debugging
        data['idx'] = idx
        return data
    
    def prefetch_batch(self, indices: list[int]) -> None:
        """
        Queue next batch items for prefetch (non-blocking).
        
        Call this with next batch indices to prefetch ahead of training.
        
        Example in training loop:
        ```python
        next_indices = list(range(batch_idx * batch_size, (batch_idx + 1) * batch_size))
        dataset.prefetch_batch(next_indices)
        ```
        """
        if not self._prefetcher:
            return
        
        for idx in indices:
            cache_key = f"item_{idx:08d}"
            self._prefetcher.queue_item(cache_key)
    
    def start_prefetch_worker(self) -> None:
        """Start async prefetch worker (optional)."""
        if self._prefetcher:
            self._prefetcher.start()
            logger.info("Prefetch worker started")
    
    def stop_prefetch_worker(self) -> None:
        """Stop async prefetch worker."""
        if self._prefetcher:
            self._prefetcher.stop()
            logger.info("Prefetch worker stopped")
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics for monitoring."""
        return self._cache.get_cache_stats()
    
    def reset_metrics(self) -> None:
        """Reset cache metrics."""
        self._cache.reset_metrics()
    
    def clear_cache(self) -> None:
        """Clear entire cache (careful with memory!)."""
        self._cache.clear()
        logger.info("Cache cleared")
    
    # ====== Private Methods ======
    
    def _init_sources(self) -> None:
        """Initialize all data sources."""
        for source_config in self.sources:
            try:
                source = create_data_source(source_config)
                self._data_sources[source_config.name] = source
                logger.info(f"Initialized data source: {source_config.name} ({source_config.source_type})")
            except Exception as e:
                logger.error(f"Failed to initialize source {source_config.name}: {e}")
                raise
    
    def _discover_dataset_size(self) -> int:
        """
        Discover total number of samples by checking first source.
        
        Assumes all samples across sources are aligned with same count.
        """
        if not self._data_sources:
            raise ValueError("No data sources initialized")
        
        # Use first source as reference
        first_source = next(iter(self._data_sources.values()))
        files = first_source.list_files()
        
        logger.info(f"Discovered {len(files)} total samples")
        return len(files)
    
    async def _fetch_item(self, cache_key: str) -> dict:
        """
        Fetch item from cache or data sources.
        
        Args:
            cache_key: Key in format "item_XXXXXXXX" where XXXXXXXX is zero-padded index
            
        Returns:
            Dict with mapped data from all sources
        """
        # Check cache first
        cached = await self._cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Extract index from cache key
        idx = int(cache_key.split('_')[1])
        
        # Fetch from sources in parallel
        result = {}
        tasks = []
        
        for source_name, source in self._data_sources.items():
            output_key = self.data_mapping.get(source_name, source_name)
            tasks.append((output_key, self._fetch_from_source(source, idx)))
        
        # Wait for all fetches
        for output_key, task in tasks:
            try:
                data = await task
                result[output_key] = data
            except Exception as e:
                logger.error(f"Failed to fetch {output_key} for idx={idx}: {e}")
                raise
        
        # Cache result
        await self._cache.put(cache_key, result, compress=True)
        
        return result
    
    async def _fetch_from_source(self, source: DataSource, idx: int) -> dict:
        """Fetch data from a specific source."""
        files = source.list_files()
        
        if idx >= len(files):
            raise IndexError(f"Index {idx} out of range for source {source.name}")
        
        file_path = files[idx]
        return await source.fetch_file(file_path, decompress=True)


# ====== Integration Helper ======

def create_streaming_dataset_from_config(config: dict) -> StreamingDatasetAdapter:
    """
    Helper to create StreamingDatasetAdapter from config dict.
    
    Example config:
    ```python
    config = {
        'sources': [
            {'name': 'local', 'type': 'local', 'path': '/data/precomputed'},
            {'name': 'hf_backup', 'type': 'huggingface', 'path': 'username/dataset'},
        ],
        'data_mapping': {
            'latents': 'latent_conditions',
            'conditions': 'text_conditions',
        },
        'cache_size_gb': 100,
        'prefetch_ahead': 10,
    }
    ```
    """
    # Convert source dicts to DataSourceConfig
    sources = [
        DataSourceConfig(
            name=s['name'],
            source_type=s['type'],
            path_or_uri=s['path'],
            credentials=s.get('credentials'),
        )
        for s in config['sources']
    ]
    
    return StreamingDatasetAdapter(
        sources=sources,
        data_mapping=config.get('data_mapping'),
        cache_size_gb=config.get('cache_size_gb', 100),
        max_ttl_seconds=config.get('max_ttl_seconds', 3600),
        prefetch_ahead=config.get('prefetch_ahead', 10),
        enable_prefetch=config.get('enable_prefetch', True),
        compression_type=config.get('compression_type', 'zstd'),
    )
