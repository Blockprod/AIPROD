"""
AIPROD Python SDK Client
=========================

Thin Python SDK for the AIPROD Video Generation API.

Usage:
    from aiprod_pipelines.api.sdk import AIPRODClient

    client = AIPRODClient(api_key="ak-xxx", base_url="https://api.aiprod.ai")
    job = client.generate("A sunset over the ocean", duration_sec=5.0)
    result = client.wait(job.job_id, poll_interval=2.0)
    print(result.video_url)
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


@dataclass
class JobResponse:
    """Response from /v1/generate or /v1/jobs/{id}."""

    job_id: str = ""
    status: str = ""  # queued | processing | completed | failed
    estimated_time_sec: Optional[float] = None
    video_url: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobResponse":
        return cls(
            job_id=data.get("job_id", ""),
            status=data.get("status", ""),
            estimated_time_sec=data.get("estimated_time_sec"),
            video_url=data.get("video_url"),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class HealthResponse:
    """Response from /health."""

    status: str = ""
    version: str = ""
    gpu_available: bool = False
    uptime_sec: float = 0.0
    active_jobs: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HealthResponse":
        return cls(
            status=data.get("status", ""),
            version=data.get("version", ""),
            gpu_available=data.get("gpu_available", False),
            uptime_sec=data.get("uptime_sec", 0.0),
            active_jobs=data.get("active_jobs", 0),
        )


# ---------------------------------------------------------------------------
# SDK Exceptions
# ---------------------------------------------------------------------------


class AIPRODError(Exception):
    """Base SDK exception."""

    def __init__(self, message: str, status_code: int = 0, body: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}


class AuthError(AIPRODError):
    pass


class RateLimitError(AIPRODError):
    pass


class ValidationError(AIPRODError):
    pass


class NotFoundError(AIPRODError):
    pass


class TimeoutError(AIPRODError):
    pass


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class AIPRODClient:
    """
    Python SDK client for the AIPROD Video Generation API.

    Supports both API key and Bearer token authentication.

    Example:
        client = AIPRODClient(api_key="ak-xxxxxxxx")
        job = client.generate("A cinematic sunset", duration_sec=10.0, width=1920, height=1080)
        result = client.wait(job.job_id)
        print(result.video_url)
    """

    DEFAULT_BASE_URL = "https://api.aiprod.ai"
    API_VERSION = "v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        bearer_token: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the AIPROD client.

        Args:
            api_key: API key for authentication (X-API-Key header).
            bearer_token: Bearer token for authentication (Authorization header).
            base_url: API base URL (default: https://api.aiprod.ai).
            timeout: Default HTTP request timeout in seconds.
        """
        if not api_key and not bearer_token:
            raise AIPRODError("Provide either api_key or bearer_token")
        self._api_key = api_key
        self._bearer_token = bearer_token
        self._base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self._timeout = timeout

    # ---- Public API ---------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        duration_sec: float = 5.0,
        width: int = 768,
        height: int = 512,
        fps: int = 24,
        seed: Optional[int] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        output_format: str = "mp4",
        hdr: bool = False,
        tts_enabled: bool = False,
        tts_text: Optional[str] = None,
        speaker_id: Optional[int] = None,
        priority: int = 0,
    ) -> JobResponse:
        """
        Submit a video generation job.

        Returns a JobResponse with job_id and initial status.
        """
        body = {
            "prompt": prompt,
            "duration_sec": duration_sec,
            "width": width,
            "height": height,
            "fps": fps,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "output_format": output_format,
            "hdr": hdr,
            "tts_enabled": tts_enabled,
            "priority": priority,
        }
        if seed is not None:
            body["seed"] = seed
        if tts_text is not None:
            body["tts_text"] = tts_text
        if speaker_id is not None:
            body["speaker_id"] = speaker_id

        data = self._post(f"/{self.API_VERSION}/generate", body)
        return JobResponse.from_dict(data)

    def get_job(self, job_id: str) -> JobResponse:
        """Get the status of a generation job."""
        data = self._get(f"/{self.API_VERSION}/jobs/{job_id}")
        return JobResponse.from_dict(data)

    def wait(
        self,
        job_id: str,
        poll_interval: float = 2.0,
        max_wait: float = 600.0,
    ) -> JobResponse:
        """
        Poll until a job reaches a terminal state (completed / failed).

        Args:
            job_id: Job ID to poll.
            poll_interval: Seconds between polls.
            max_wait: Maximum total wait time in seconds.

        Returns:
            Final JobResponse.

        Raises:
            TimeoutError: If max_wait is exceeded.
        """
        start = time.time()
        while True:
            job = self.get_job(job_id)
            if job.status in ("completed", "failed"):
                return job
            elapsed = time.time() - start
            if elapsed >= max_wait:
                raise TimeoutError(
                    f"Job {job_id} did not complete within {max_wait}s (status: {job.status})"
                )
            time.sleep(poll_interval)

    def health(self) -> HealthResponse:
        """Check API health."""
        data = self._get("/health")
        return HealthResponse.from_dict(data)

    def list_jobs(self, limit: int = 20, offset: int = 0) -> List[JobResponse]:
        """List recent jobs (if supported by the API)."""
        data = self._get(f"/{self.API_VERSION}/jobs?limit={limit}&offset={offset}")
        if isinstance(data, list):
            return [JobResponse.from_dict(j) for j in data]
        jobs = data.get("jobs", [])
        return [JobResponse.from_dict(j) for j in jobs]

    # ---- HTTP internals -----------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Request-ID": str(uuid.uuid4()),
            "User-Agent": "aiprod-python-sdk/1.0.0",
        }
        if self._api_key:
            headers["X-API-Key"] = self._api_key
        if self._bearer_token:
            headers["Authorization"] = f"Bearer {self._bearer_token}"
        return headers

    def _get(self, path: str) -> Dict[str, Any]:
        url = f"{self._base_url}{path}"
        req = Request(url, headers=self._headers(), method="GET")
        return self._do(req)

    def _post(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self._base_url}{path}"
        data = json.dumps(body).encode()
        req = Request(url, data=data, headers=self._headers(), method="POST")
        return self._do(req)

    def _do(self, req: Request) -> Dict[str, Any]:
        """Execute an HTTP request and handle errors."""
        try:
            resp = urlopen(req, timeout=self._timeout)
            body = json.loads(resp.read().decode())
            return body
        except HTTPError as e:
            body_text = e.read().decode() if e.fp else "{}"
            try:
                error_body = json.loads(body_text)
            except json.JSONDecodeError:
                error_body = {"error": body_text}
            msg = error_body.get("error", f"HTTP {e.code}")
            exc_map = {401: AuthError, 429: RateLimitError, 422: ValidationError, 404: NotFoundError}
            exc_cls = exc_map.get(e.code, AIPRODError)
            raise exc_cls(msg, status_code=e.code, body=error_body) from e
        except URLError as e:
            raise AIPRODError(f"Connection error: {e}") from e


# ---------------------------------------------------------------------------
# JavaScript SDK stub (for documentation / generation)
# ---------------------------------------------------------------------------

JAVASCRIPT_SDK_EXAMPLE = """
// AIPROD JavaScript SDK — Usage example
// npm install aiprod-sdk

import { AIPRODClient } from 'aiprod-sdk';

const client = new AIPRODClient({ apiKey: 'ak-xxxxxxxx' });

const job = await client.generate({
  prompt: 'A cinematic sunset over the ocean',
  durationSec: 10.0,
  width: 1920,
  height: 1080,
});

const result = await client.waitForCompletion(job.jobId);
console.log(result.videoUrl);
"""
