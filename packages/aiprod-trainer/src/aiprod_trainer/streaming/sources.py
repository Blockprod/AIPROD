"""
Data source abstractions for streaming from multiple backends (Local, HF, S3, GCS).
Enables unified interface for fetching compressed latent and condition files.
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


class HuggingFaceDataSource(DataSource):
    """Load data from Hugging Face Hub."""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        from huggingface_hub import hf_hub_download
        self.hf_hub_download = hf_hub_download
        self.repo_id = config.path_or_uri  # e.g. "username/dataset-repo"
        self.cache_dir = config.credentials.get('cache_dir', None) if config.credentials else None
    
    async def fetch_file(self, file_path: str, decompress: bool = True) -> dict:
        """Fetch file from HF Hub."""
        loop = asyncio.get_event_loop()
        file_path_cached = await loop.run_in_executor(
            None,
            lambda: self.hf_hub_download(
                repo_id=self.repo_id,
                filename=file_path,
                cache_dir=self.cache_dir,
                repo_type='dataset'
            )
        )
        
        # Load the cached file
        import zstandard as zstd
        if file_path.endswith('.zst') and decompress:
            with open(file_path_cached, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                decompressed = dctx.stream_reader(f).read()
            import io
            return torch.load(io.BytesIO(decompressed), map_location='cpu', weights_only=True)
        else:
            return torch.load(file_path_cached, map_location='cpu', weights_only=True)
    
    def list_files(self, directory: str = "") -> list[str]:
        """List files from HF Hub."""
        from huggingface_hub import list_files_in_repo
        
        files = list_files_in_repo(
            repo_id=self.repo_id,
            repo_type='dataset',
            recursive=True,
        )
        
        # Filter to .pt and .pt.zst files
        return [f for f in files if f.endswith('.pt') or f.endswith('.pt.zst')]
    
    async def prefetch_files(self, file_paths: list[str]) -> None:
        """Prefetch multiple files from HF."""
        tasks = [self.fetch_file(path, decompress=False) for path in file_paths]
        await asyncio.gather(*tasks, return_exceptions=True)


class S3DataSource(DataSource):
    """Load data from AWS S3."""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        import boto3
        self.s3_client = boto3.client('s3')
        
        # Parse s3://bucket/prefix from config.path_or_uri
        s3_uri = config.path_or_uri
        if s3_uri.startswith('s3://'):
            parts = s3_uri[5:].split('/', 1)
            self.bucket = parts[0]
            self.prefix = parts[1] if len(parts) > 1 else ""
        else:
            raise ValueError(f"Invalid S3 URI: {s3_uri}")
    
    async def fetch_file(self, file_path: str, decompress: bool = True) -> dict:
        """Fetch file from S3."""
        loop = asyncio.get_event_loop()
        
        s3_key = f"{self.prefix}/{file_path}".lstrip('/')
        
        data = await loop.run_in_executor(
            None,
            lambda: self._load_from_s3(s3_key, decompress)
        )
        return data
    
    def _load_from_s3(self, s3_key: str, decompress: bool) -> dict:
        """Load object from S3 (runs in executor)."""
        import zstandard as zstd
        import io
        
        response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
        data_bytes = response['Body'].read()
        
        if s3_key.endswith('.zst') and decompress:
            dctx = zstd.ZstdDecompressor()
            data_bytes = dctx.decompress(data_bytes)
        
        return torch.load(io.BytesIO(data_bytes), map_location='cpu', weights_only=True)
    
    def list_files(self, directory: str = "") -> list[str]:
        """List files from S3."""
        prefix = f"{self.prefix}/{directory}".lstrip('/')
        files = []
        
        paginator = self.s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('.pt') or key.endswith('.pt.zst'):
                    # Return relative path from prefix
                    files.append(key[len(self.prefix):].lstrip('/'))
        
        return sorted(files)
    
    async def prefetch_files(self, file_paths: list[str]) -> None:
        """Prefetch multiple files from S3."""
        tasks = [self.fetch_file(path, decompress=False) for path in file_paths]
        await asyncio.gather(*tasks, return_exceptions=True)


class GCSDataSource(DataSource):
    """Load data from Google Cloud Storage."""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        from google.cloud import storage
        self.gcs_client = storage.Client()
        
        # Parse gs://bucket/prefix from config.path_or_uri
        gcs_uri = config.path_or_uri
        if gcs_uri.startswith('gs://'):
            parts = gcs_uri[5:].split('/', 1)
            self.bucket_name = parts[0]
            self.prefix = parts[1] if len(parts) > 1 else ""
        else:
            raise ValueError(f"Invalid GCS URI: {gcs_uri}")
        
        self.bucket = self.gcs_client.bucket(self.bucket_name)
    
    async def fetch_file(self, file_path: str, decompress: bool = True) -> dict:
        """Fetch file from GCS."""
        loop = asyncio.get_event_loop()
        
        gcs_path = f"{self.prefix}/{file_path}".lstrip('/')
        
        data = await loop.run_in_executor(
            None,
            lambda: self._load_from_gcs(gcs_path, decompress)
        )
        return data
    
    def _load_from_gcs(self, gcs_path: str, decompress: bool) -> dict:
        """Load object from GCS (runs in executor)."""
        import zstandard as zstd
        import io
        
        blob = self.bucket.blob(gcs_path)
        data_bytes = blob.download_as_bytes()
        
        if gcs_path.endswith('.zst') and decompress:
            dctx = zstd.ZstdDecompressor()
            data_bytes = dctx.decompress(data_bytes)
        
        return torch.load(io.BytesIO(data_bytes), map_location='cpu', weights_only=True)
    
    def list_files(self, directory: str = "") -> list[str]:
        """List files from GCS."""
        prefix = f"{self.prefix}/{directory}".lstrip('/')
        files = []
        
        for blob in self.gcs_client.list_blobs(self.bucket_name, prefix=prefix):
            if blob.name.endswith('.pt') or blob.name.endswith('.pt.zst'):
                # Return relative path from prefix
                files.append(blob.name[len(self.prefix):].lstrip('/'))
        
        return sorted(files)
    
    async def prefetch_files(self, file_paths: list[str]) -> None:
        """Prefetch multiple files from GCS."""
        tasks = [self.fetch_file(path, decompress=False) for path in file_paths]
        await asyncio.gather(*tasks, return_exceptions=True)


def create_data_source(config: DataSourceConfig) -> DataSource:
    """Factory function to create appropriate data source."""
    source_mapping = {
        'local': LocalDataSource,
        'huggingface': HuggingFaceDataSource,
        's3': S3DataSource,
        'gcs': GCSDataSource,
    }
    
    source_class = source_mapping.get(config.source_type)
    if not source_class:
        raise ValueError(f"Unknown source type: {config.source_type}")
    
    return source_class(config)
