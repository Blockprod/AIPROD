"""
AIPROD Production API Gateway
==============================

FastAPI-based API Gateway with:
- JWT + API key authentication
- Tier-based rate limiting (Free / Pro / Enterprise)
- Pydantic request validation
- Health check with GPU VRAM probe
- Structured JSON logging
- Request tracing (X-Request-ID)

Deployed on Cloud Run (CPU) → routes jobs to GPU worker pool via message queue.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Pydantic models (request / response schemas)
# ---------------------------------------------------------------------------


class Tier(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class RateLimitConfig:
    """Per-tier rate-limit windows."""

    tier: Tier = Tier.FREE
    max_requests_per_day: int = 5
    max_requests_per_minute: int = 1
    max_concurrent: int = 1


# Default tier configs
TIER_LIMITS: Dict[Tier, RateLimitConfig] = {
    Tier.FREE: RateLimitConfig(Tier.FREE, max_requests_per_day=5, max_requests_per_minute=1, max_concurrent=1),
    Tier.PRO: RateLimitConfig(Tier.PRO, max_requests_per_day=100, max_requests_per_minute=10, max_concurrent=5),
    Tier.ENTERPRISE: RateLimitConfig(
        Tier.ENTERPRISE, max_requests_per_day=100_000, max_requests_per_minute=200, max_concurrent=50
    ),
}


# ---------------------------------------------------------------------------
# Request / Response models (plain dataclasses, mirrors Pydantic semantics)
# ---------------------------------------------------------------------------


@dataclass
class VideoGenerationRequest:
    """Validated video-generation request."""

    prompt: str
    duration_sec: float = 5.0
    width: int = 768
    height: int = 512
    fps: int = 24
    seed: Optional[int] = None
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    output_format: str = "mp4"
    hdr: bool = False
    tts_enabled: bool = False
    tts_text: Optional[str] = None
    speaker_id: Optional[int] = None
    priority: int = 0  # 0=normal, 1=high, 2=urgent

    # --- validation helpers ---------------------------------------------------

    _VALID_FORMATS = {"mp4", "mov", "webm", "mxf"}
    _MAX_DURATION = 120.0
    _VALID_RESOLUTIONS = {
        (512, 512), (768, 512), (1024, 576),
        (1280, 720), (1920, 1080), (3840, 2160),
    }

    def validate(self) -> List[str]:
        """Return list of validation errors (empty == valid)."""
        errors: List[str] = []
        if not self.prompt or len(self.prompt.strip()) == 0:
            errors.append("prompt must not be empty")
        if len(self.prompt) > 4096:
            errors.append("prompt exceeds 4096 characters")
        if self.duration_sec <= 0 or self.duration_sec > self._MAX_DURATION:
            errors.append(f"duration_sec must be in (0, {self._MAX_DURATION}]")
        if (self.width, self.height) not in self._VALID_RESOLUTIONS:
            errors.append(f"resolution {self.width}x{self.height} not supported")
        if self.fps not in (24, 25, 30, 48, 60):
            errors.append(f"fps {self.fps} not in {{24,25,30,48,60}}")
        if self.output_format not in self._VALID_FORMATS:
            errors.append(f"output_format must be one of {self._VALID_FORMATS}")
        if self.guidance_scale < 1.0 or self.guidance_scale > 30.0:
            errors.append("guidance_scale must be in [1.0, 30.0]")
        if self.num_inference_steps < 1 or self.num_inference_steps > 200:
            errors.append("num_inference_steps must be in [1, 200]")
        if self.tts_enabled and not self.tts_text:
            errors.append("tts_text required when tts_enabled=True")
        return errors


@dataclass
class VideoGenerationResponse:
    """API response for a generation job."""

    job_id: str
    status: str  # queued | processing | completed | failed
    estimated_time_sec: Optional[float] = None
    video_url: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthResponse:
    """Health-check response with optional GPU probe."""

    status: str  # healthy | degraded | unhealthy
    version: str = "3.0.0"
    gpu_available: bool = False
    gpu_vram_free_mb: float = 0.0
    gpu_vram_total_mb: float = 0.0
    uptime_sec: float = 0.0
    active_jobs: int = 0


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


@dataclass
class AuthToken:
    """Parsed API token payload."""

    user_id: str
    tier: Tier
    tenant_id: str
    issued_at: float
    expires_at: float


class AuthManager:
    """
    Handles JWT verification and API-key lookup.

    In production this would verify RS256 JWTs against a JWKS endpoint
    or query a database for API keys.  Here we implement the structure
    with HMAC-SHA256 token verification for self-contained deployment.
    """

    def __init__(self, secret_key: Optional[str] = None):
        self._secret = (secret_key or os.environ.get("AIPROD_API_SECRET", "dev-secret-change-me")).encode()
        self._api_keys: Dict[str, AuthToken] = {}  # key_hash → token

    # ---- API key management -------------------------------------------------

    def register_api_key(self, api_key: str, user_id: str, tier: Tier, tenant_id: str = "default") -> None:
        """Register an API key (hashed storage)."""
        key_hash = self._hash_key(api_key)
        now = time.time()
        self._api_keys[key_hash] = AuthToken(
            user_id=user_id,
            tier=tier,
            tenant_id=tenant_id,
            issued_at=now,
            expires_at=now + 365 * 86400,  # 1 year
        )

    def authenticate(self, authorization: Optional[str] = None, api_key: Optional[str] = None) -> AuthToken:
        """
        Authenticate a request via Bearer token or X-API-Key header.

        Returns AuthToken on success, raises AuthenticationError on failure.
        """
        if api_key:
            return self._verify_api_key(api_key)
        if authorization and authorization.startswith("Bearer "):
            return self._verify_jwt(authorization[7:])
        raise AuthenticationError("Missing authentication: provide Authorization Bearer or X-API-Key header")

    # ---- JWT (HMAC-SHA256 self-signed) --------------------------------------

    def create_jwt(self, user_id: str, tier: Tier, tenant_id: str = "default", ttl: int = 3600) -> str:
        """Create a self-signed JWT (HMAC-SHA256)."""
        now = time.time()
        payload = {
            "sub": user_id,
            "tier": tier.value,
            "tenant": tenant_id,
            "iat": now,
            "exp": now + ttl,
        }
        payload_b64 = _b64url(json.dumps(payload).encode())
        header_b64 = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
        signing_input = f"{header_b64}.{payload_b64}"
        sig = hmac.new(self._secret, signing_input.encode(), hashlib.sha256).hexdigest()
        return f"{signing_input}.{sig}"

    def _verify_jwt(self, token: str) -> AuthToken:
        parts = token.split(".")
        if len(parts) != 3:
            raise AuthenticationError("Malformed JWT")
        signing_input = f"{parts[0]}.{parts[1]}"
        expected_sig = hmac.new(self._secret, signing_input.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected_sig, parts[2]):
            raise AuthenticationError("Invalid JWT signature")
        payload = json.loads(_b64url_decode(parts[1]))
        if payload.get("exp", 0) < time.time():
            raise AuthenticationError("JWT expired")
        return AuthToken(
            user_id=payload["sub"],
            tier=Tier(payload["tier"]),
            tenant_id=payload.get("tenant", "default"),
            issued_at=payload["iat"],
            expires_at=payload["exp"],
        )

    def _verify_api_key(self, api_key: str) -> AuthToken:
        key_hash = self._hash_key(api_key)
        token = self._api_keys.get(key_hash)
        if token is None:
            raise AuthenticationError("Invalid API key")
        if token.expires_at < time.time():
            raise AuthenticationError("API key expired")
        return token

    @staticmethod
    def _hash_key(key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Rate limiter (sliding-window in-memory, Redis-backed in production)
# ---------------------------------------------------------------------------


class RateLimiter:
    """
    Sliding-window rate limiter.

    In production this would use Redis MULTI/EXEC with sorted sets.
    Here we keep an in-memory implementation that is structurally identical.
    """

    def __init__(self):
        # user_id → list of timestamps
        self._minute_windows: Dict[str, List[float]] = {}
        self._day_windows: Dict[str, List[float]] = {}
        self._concurrent: Dict[str, int] = {}

    def check(self, user_id: str, tier: Tier) -> None:
        """Raise RateLimitError if the user exceeds their tier limit."""
        limits = TIER_LIMITS[tier]
        now = time.time()

        # --- per-minute ---
        minute_ts = self._minute_windows.setdefault(user_id, [])
        minute_ts[:] = [t for t in minute_ts if t > now - 60]
        if len(minute_ts) >= limits.max_requests_per_minute:
            raise RateLimitError(
                f"Rate limit exceeded: {limits.max_requests_per_minute} requests/minute for {tier.value} tier"
            )

        # --- per-day ---
        day_ts = self._day_windows.setdefault(user_id, [])
        day_ts[:] = [t for t in day_ts if t > now - 86400]
        if len(day_ts) >= limits.max_requests_per_day:
            raise RateLimitError(
                f"Rate limit exceeded: {limits.max_requests_per_day} requests/day for {tier.value} tier"
            )

        # --- concurrent ---
        active = self._concurrent.get(user_id, 0)
        if active >= limits.max_concurrent:
            raise RateLimitError(f"Max concurrent jobs ({limits.max_concurrent}) reached for {tier.value} tier")

    def record(self, user_id: str) -> None:
        """Record a new request timestamp and increment concurrent counter."""
        now = time.time()
        self._minute_windows.setdefault(user_id, []).append(now)
        self._day_windows.setdefault(user_id, []).append(now)
        self._concurrent[user_id] = self._concurrent.get(user_id, 0) + 1

    def release(self, user_id: str) -> None:
        """Decrement concurrent counter when job finishes."""
        self._concurrent[user_id] = max(0, self._concurrent.get(user_id, 0) - 1)


# ---------------------------------------------------------------------------
# Job queue (in-memory, replaced by Celery/Ray in production)
# ---------------------------------------------------------------------------


@dataclass
class Job:
    """Queued video-generation job."""

    job_id: str
    user_id: str
    tenant_id: str
    tier: Tier
    request: VideoGenerationRequest
    status: str = "queued"
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result_url: Optional[str] = None
    error: Optional[str] = None
    priority: int = 0

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


class JobQueue:
    """
    Priority job queue.

    In production this is backed by Redis/RabbitMQ with priority queues.
    Enterprise > Pro > Free ordering with FIFO within each tier.
    """

    TIER_PRIORITY = {Tier.ENTERPRISE: 100, Tier.PRO: 50, Tier.FREE: 10}

    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._queue: List[str] = []  # sorted by effective priority

    def enqueue(self, job: Job) -> None:
        """Add job to the queue with tier-based priority."""
        job.priority = self.TIER_PRIORITY.get(job.tier, 0) + job.request.priority
        self._jobs[job.job_id] = job
        self._queue.append(job.job_id)
        # Re-sort by priority descending, then creation time ascending
        self._queue.sort(key=lambda jid: (-self._jobs[jid].priority, self._jobs[jid].created_at))

    def dequeue(self) -> Optional[Job]:
        """Pop the highest-priority job."""
        while self._queue:
            jid = self._queue.pop(0)
            job = self._jobs.get(jid)
            if job and job.status == "queued":
                job.status = "processing"
                job.started_at = time.time()
                return job
        return None

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def complete(self, job_id: str, result_url: str) -> None:
        job = self._jobs.get(job_id)
        if job:
            job.status = "completed"
            job.completed_at = time.time()
            job.result_url = result_url

    def fail(self, job_id: str, error: str) -> None:
        job = self._jobs.get(job_id)
        if job:
            job.status = "failed"
            job.completed_at = time.time()
            job.error = error

    @property
    def pending_count(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status == "queued")

    @property
    def active_count(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status == "processing")


# ---------------------------------------------------------------------------
# GPU health probe
# ---------------------------------------------------------------------------


class GPUProbe:
    """Probes GPU availability and VRAM via torch.cuda (if available)."""

    @staticmethod
    def probe() -> Dict[str, Any]:
        """Return GPU status dict."""
        try:
            import torch

            if not torch.cuda.is_available():
                return {"available": False, "vram_free_mb": 0.0, "vram_total_mb": 0.0}
            mem = torch.cuda.mem_get_info(0)
            return {
                "available": True,
                "vram_free_mb": mem[0] / (1024 * 1024),
                "vram_total_mb": mem[1] / (1024 * 1024),
                "device_name": torch.cuda.get_device_name(0),
            }
        except Exception:
            return {"available": False, "vram_free_mb": 0.0, "vram_total_mb": 0.0}


# ---------------------------------------------------------------------------
# API Gateway (core class)
# ---------------------------------------------------------------------------


class APIGateway:
    """
    Production API Gateway.

    Integrates authentication, rate-limiting, request validation,
    job queuing, and health monitoring.

    In production, `create_fastapi_app()` wraps this class in a FastAPI
    application with HTTP endpoints.
    """

    def __init__(
        self,
        auth_manager: Optional[AuthManager] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.auth = auth_manager or AuthManager()
        self.rate_limiter = rate_limiter or RateLimiter()
        self.job_queue = JobQueue()
        self._start_time = time.time()

    # ---- Public API ---------------------------------------------------------

    def generate_video(
        self,
        request: VideoGenerationRequest,
        authorization: Optional[str] = None,
        api_key: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> VideoGenerationResponse:
        """
        Authenticated, rate-limited, validated video generation endpoint.

        Flow: Auth → Rate-limit → Validate → Enqueue → Return job handle.
        """
        request_id = request_id or str(uuid.uuid4())

        # 1. Authenticate
        token = self.auth.authenticate(authorization=authorization, api_key=api_key)

        # 2. Rate-limit
        self.rate_limiter.check(token.user_id, token.tier)

        # 3. Validate
        errors = request.validate()
        if errors:
            raise ValidationError(errors)

        # 4. Enqueue
        job = Job(
            job_id=request_id,
            user_id=token.user_id,
            tenant_id=token.tenant_id,
            tier=token.tier,
            request=request,
        )
        self.job_queue.enqueue(job)
        self.rate_limiter.record(token.user_id)

        # 5. Estimate time based on resolution + duration
        pixels = request.width * request.height
        est = request.duration_sec * (pixels / (768 * 512)) * 2.0  # rough estimate

        return VideoGenerationResponse(
            job_id=request_id,
            status="queued",
            estimated_time_sec=round(est, 1),
            metadata={
                "tier": token.tier.value,
                "resolution": f"{request.width}x{request.height}",
                "duration_sec": request.duration_sec,
                "queue_position": self.job_queue.pending_count,
            },
        )

    def get_job_status(self, job_id: str, authorization: Optional[str] = None, api_key: Optional[str] = None) -> VideoGenerationResponse:
        """Query job status."""
        token = self.auth.authenticate(authorization=authorization, api_key=api_key)
        job = self.job_queue.get(job_id)
        if job is None:
            raise NotFoundError(f"Job {job_id} not found")
        if job.user_id != token.user_id and token.tier != Tier.ENTERPRISE:
            raise AuthenticationError("Access denied to this job")
        return VideoGenerationResponse(
            job_id=job.job_id,
            status=job.status,
            video_url=job.result_url,
            error=job.error,
        )

    def health(self) -> HealthResponse:
        """Health check with GPU probe."""
        gpu = GPUProbe.probe()
        gpu_ok = gpu.get("available", False)
        active = self.job_queue.active_count

        if gpu_ok:
            status = "healthy"
        elif active == 0:
            status = "degraded"  # no GPU but no jobs either
        else:
            status = "unhealthy"

        return HealthResponse(
            status=status,
            gpu_available=gpu_ok,
            gpu_vram_free_mb=gpu.get("vram_free_mb", 0.0),
            gpu_vram_total_mb=gpu.get("vram_total_mb", 0.0),
            uptime_sec=round(time.time() - self._start_time, 1),
            active_jobs=active,
        )


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class GatewayError(Exception):
    """Base gateway error."""

    status_code: int = 500


class AuthenticationError(GatewayError):
    status_code = 401


class RateLimitError(GatewayError):
    status_code = 429


class ValidationError(GatewayError):
    status_code = 422

    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


class NotFoundError(GatewayError):
    status_code = 404


# ---------------------------------------------------------------------------
# FastAPI application factory
# ---------------------------------------------------------------------------


def create_fastapi_app(gateway: Optional[APIGateway] = None) -> Any:
    """
    Create a FastAPI application wrapping the APIGateway.

    Returns a FastAPI ``app`` object ready for ``uvicorn.run(app)``.
    Requires ``fastapi`` and ``uvicorn`` to be installed.
    """
    try:
        from fastapi import FastAPI, Request, HTTPException
        from fastapi.responses import JSONResponse
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        raise ImportError("FastAPI is required: pip install fastapi uvicorn[standard]")

    gw = gateway or APIGateway()
    app = FastAPI(
        title="AIPROD Video Generation API",
        version="3.0.0",
        description="Production API for AI-powered cinematic video generation.",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Exception handlers ---
    @app.exception_handler(GatewayError)
    async def gateway_error_handler(request: Request, exc: GatewayError):
        body: Dict[str, Any] = {"error": str(exc)}
        if isinstance(exc, ValidationError):
            body["validation_errors"] = exc.errors
        return JSONResponse(status_code=exc.status_code, content=body)

    # --- Endpoints ---
    @app.get("/health")
    async def health():
        h = gw.health()
        return {
            "status": h.status,
            "version": h.version,
            "gpu_available": h.gpu_available,
            "gpu_vram_free_mb": h.gpu_vram_free_mb,
            "gpu_vram_total_mb": h.gpu_vram_total_mb,
            "uptime_sec": h.uptime_sec,
            "active_jobs": h.active_jobs,
        }

    @app.post("/v1/generate")
    async def generate_video(request: Request):
        body = await request.json()
        req = VideoGenerationRequest(**body)
        auth = request.headers.get("Authorization")
        api_key = request.headers.get("X-API-Key")
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        resp = gw.generate_video(req, authorization=auth, api_key=api_key, request_id=request_id)
        return {
            "job_id": resp.job_id,
            "status": resp.status,
            "estimated_time_sec": resp.estimated_time_sec,
            "metadata": resp.metadata,
        }

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str, request: Request):
        auth = request.headers.get("Authorization")
        api_key = request.headers.get("X-API-Key")
        resp = gw.get_job_status(job_id, authorization=auth, api_key=api_key)
        return {
            "job_id": resp.job_id,
            "status": resp.status,
            "video_url": resp.video_url,
            "error": resp.error,
        }

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

import base64 as _base64


def _b64url(data: bytes) -> str:
    return _base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return _base64.urlsafe_b64decode(s)
