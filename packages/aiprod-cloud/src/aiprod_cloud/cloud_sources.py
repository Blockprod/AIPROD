"""
Cloud Streaming Data Sources — S3 / GCS / HuggingFace Hub.
============================================================

These data-source classes require cloud SDK dependencies and live in
``aiprod-cloud``.   The sovereign ``LocalDataSource`` stays in
``aiprod_trainer.streaming.sources``.

Re-exported by the backward-compatible shim in
``aiprod_trainer.streaming.sources`` (when ``aiprod-cloud`` is installed).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, List

import torch

from aiprod_trainer.streaming.sources import DataSource, DataSourceConfig


# ---------------------------------------------------------------------------
# HuggingFace Hub
# ---------------------------------------------------------------------------


class HuggingFaceDataSource(DataSource):
    """Load data from Hugging Face Hub.

    Requires optional dependency: pip install huggingface-hub
    """

    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        self.hf_hub_download = hf_hub_download
        self.repo_id = config.path_or_uri
        self.cache_dir = config.credentials.get("cache_dir", None) if config.credentials else None

    async def fetch_file(self, file_path: str, decompress: bool = True) -> dict:
        loop = asyncio.get_event_loop()
        file_path_cached = await loop.run_in_executor(
            None,
            lambda: self.hf_hub_download(
                repo_id=self.repo_id,
                filename=file_path,
                cache_dir=self.cache_dir,
                repo_type="dataset",
            ),
        )

        import zstandard as zstd  # noqa: PLC0415
        if file_path.endswith(".zst") and decompress:
            with open(file_path_cached, "rb") as f:
                dctx = zstd.ZstdDecompressor()
                decompressed = dctx.stream_reader(f).read()
            import io  # noqa: PLC0415
            return torch.load(io.BytesIO(decompressed), map_location="cpu", weights_only=True)
        else:
            return torch.load(file_path_cached, map_location="cpu", weights_only=True)

    def list_files(self, directory: str = "") -> list[str]:
        from huggingface_hub import list_files_in_repo  # noqa: PLC0415

        files = list_files_in_repo(repo_id=self.repo_id, repo_type="dataset", recursive=True)
        return [f for f in files if f.endswith(".pt") or f.endswith(".pt.zst")]

    async def prefetch_files(self, file_paths: list[str]) -> None:
        tasks = [self.fetch_file(path, decompress=False) for path in file_paths]
        await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# AWS S3
# ---------------------------------------------------------------------------


class S3DataSource(DataSource):
    """Load data from AWS S3.

    Requires: pip install boto3 (or: pip install aiprod-cloud[s3])
    """

    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        import boto3  # type: ignore[import-not-found]  # noqa: PLC0415

        self.s3_client = boto3.client("s3")

        s3_uri = config.path_or_uri
        if s3_uri.startswith("s3://"):
            parts = s3_uri[5:].split("/", 1)
            self.bucket = parts[0]
            self.prefix = parts[1] if len(parts) > 1 else ""
        else:
            raise ValueError(f"Invalid S3 URI: {s3_uri}")

    async def fetch_file(self, file_path: str, decompress: bool = True) -> dict:
        loop = asyncio.get_event_loop()
        s3_key = f"{self.prefix}/{file_path}".lstrip("/")
        return await loop.run_in_executor(None, lambda: self._load_from_s3(s3_key, decompress))

    def _load_from_s3(self, s3_key: str, decompress: bool) -> dict:
        import io  # noqa: PLC0415
        import zstandard as zstd  # noqa: PLC0415

        response = self.s3_client.get_object(Bucket=self.bucket, Key=s3_key)
        data_bytes = response["Body"].read()

        if s3_key.endswith(".zst") and decompress:
            dctx = zstd.ZstdDecompressor()
            data_bytes = dctx.decompress(data_bytes)

        return torch.load(io.BytesIO(data_bytes), map_location="cpu", weights_only=True)

    def list_files(self, directory: str = "") -> list[str]:
        prefix = f"{self.prefix}/{directory}".lstrip("/")
        files = []
        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".pt") or key.endswith(".pt.zst"):
                    files.append(key[len(self.prefix) :].lstrip("/"))
        return sorted(files)

    async def prefetch_files(self, file_paths: list[str]) -> None:
        tasks = [self.fetch_file(path, decompress=False) for path in file_paths]
        await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------------
# Google Cloud Storage
# ---------------------------------------------------------------------------


class GCSDataSource(DataSource):
    """Load data from Google Cloud Storage.

    Requires: pip install google-cloud-storage (or: pip install aiprod-cloud[gcp])
    """

    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        from google.cloud import storage  # noqa: PLC0415

        self.gcs_client = storage.Client()

        gcs_uri = config.path_or_uri
        if gcs_uri.startswith("gs://"):
            parts = gcs_uri[5:].split("/", 1)
            self.bucket_name = parts[0]
            self.prefix = parts[1] if len(parts) > 1 else ""
        else:
            raise ValueError(f"Invalid GCS URI: {gcs_uri}")

        self.bucket = self.gcs_client.bucket(self.bucket_name)

    async def fetch_file(self, file_path: str, decompress: bool = True) -> dict:
        loop = asyncio.get_event_loop()
        gcs_path = f"{self.prefix}/{file_path}".lstrip("/")
        return await loop.run_in_executor(None, lambda: self._load_from_gcs(gcs_path, decompress))

    def _load_from_gcs(self, gcs_path: str, decompress: bool) -> dict:
        import io  # noqa: PLC0415
        import zstandard as zstd  # noqa: PLC0415

        blob = self.bucket.blob(gcs_path)
        data_bytes = blob.download_as_bytes()

        if gcs_path.endswith(".zst") and decompress:
            dctx = zstd.ZstdDecompressor()
            data_bytes = dctx.decompress(data_bytes)

        return torch.load(io.BytesIO(data_bytes), map_location="cpu", weights_only=True)

    def list_files(self, directory: str = "") -> list[str]:
        prefix = f"{self.prefix}/{directory}".lstrip("/")
        files = []
        for blob in self.gcs_client.list_blobs(self.bucket_name, prefix=prefix):
            if blob.name.endswith(".pt") or blob.name.endswith(".pt.zst"):
                files.append(blob.name[len(self.prefix) :].lstrip("/"))
        return sorted(files)

    async def prefetch_files(self, file_paths: list[str]) -> None:
        tasks = [self.fetch_file(path, decompress=False) for path in file_paths]
        await asyncio.gather(*tasks, return_exceptions=True)
