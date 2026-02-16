"""
Data source abstractions for streaming from multiple backends.

Sovereign sources (Local) live here.
Cloud sources (HF, S3, GCS) live in ``aiprod-cloud`` and are loaded
dynamically when that package is installed.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class DataSourceConfig:
    """Configuration for a data source."""
    name: str
    source_type: str  # 'local', 'huggingface', 's3', 'gcs'
    path_or_uri: str
    credentials: Optional[dict] = None


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.name = config.name
    
    @abstractmethod
    async def fetch_file(self, file_path: str, decompress: bool = True) -> dict:
        """
        Fetch a file from the source.
        
        Args:
            file_path: Relative path within source
            decompress: Whether to decompress zstd if needed
            
        Returns:
            Dictionary with loaded data
        """
        pass
    
    @abstractmethod
    def list_files(self, directory: str = "") -> list[str]:
        """List all files in source (or subdirectory)."""
        pass
    
    @abstractmethod
    async def prefetch_files(self, file_paths: list[str]) -> None:
        """Prefetch multiple files concurrently."""
        pass


class LocalDataSource(DataSource):
    """Load data from local filesystem."""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.base_path = Path(config.path_or_uri).expanduser().resolve()
        if not self.base_path.exists():
            raise FileNotFoundError(f"Local data source path does not exist: {self.base_path}")
    
    async def fetch_file(self, file_path: str, decompress: bool = True) -> dict:
        """Fetch file from local filesystem."""
        full_path = self.base_path / file_path
        
        # Support both .pt (uncompressed) and .pt.zst (compressed)
        if not full_path.exists() and (self.base_path / f"{file_path}.zst").exists():
            full_path = self.base_path / f"{file_path}.zst"
        
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {full_path}")
        
        # Load in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None,
            lambda: self._load_file(full_path, decompress)
        )
        return data
    
    @staticmethod
    def _load_file(path: Path, decompress: bool) -> dict:
        """Load file from disk (runs in executor)."""
        import zstandard as zstd
        
        if path.suffix == '.zst' and decompress:
            # Decompress .pt.zst
            with open(path, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                decompressed = dctx.stream_reader(f).read()
            import io
            return torch.load(io.BytesIO(decompressed), map_location='cpu', weights_only=True)
        else:
            # Load .pt directly
            return torch.load(path, map_location='cpu', weights_only=True)
    
    def list_files(self, directory: str = "") -> list[str]:
        """List all .pt and .pt.zst files in directory."""
        dir_path = self.base_path / directory if directory else self.base_path
        files = []
        
        for suffix in ['*.pt', '*.pt.zst']:
            files.extend([str(f.relative_to(self.base_path)) for f in dir_path.glob(f"**/{suffix}")])
        
        return sorted(files)
    
    async def prefetch_files(self, file_paths: list[str]) -> None:
        """Prefetch multiple files concurrently."""
        tasks = [self.fetch_file(path, decompress=False) for path in file_paths]
        await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# Cloud source shims — loaded from aiprod-cloud when installed
# ---------------------------------------------------------------------------

try:
    from aiprod_cloud.cloud_sources import (  # noqa: PLC0415
        GCSDataSource,
        HuggingFaceDataSource,
        S3DataSource,
    )
except ImportError:
    HuggingFaceDataSource = None  # type: ignore[assignment,misc]
    S3DataSource = None  # type: ignore[assignment,misc]
    GCSDataSource = None  # type: ignore[assignment,misc]


def create_data_source(config: DataSourceConfig) -> DataSource:
    """Factory function to create appropriate data source."""
    source_mapping: dict = {
        'local': LocalDataSource,
    }
    # Register cloud sources when aiprod-cloud is installed
    if HuggingFaceDataSource is not None:
        source_mapping['huggingface'] = HuggingFaceDataSource
    if S3DataSource is not None:
        source_mapping['s3'] = S3DataSource
    if GCSDataSource is not None:
        source_mapping['gcs'] = GCSDataSource

    source_class = source_mapping.get(config.source_type)
    if not source_class:
        available = ', '.join(sorted(source_mapping.keys()))
        raise ValueError(
            f"Unknown source type: {config.source_type}. "
            f"Available: {available}. "
            f"Install aiprod-cloud for cloud sources (s3, gcs, huggingface)."
        )

    return source_class(config)
