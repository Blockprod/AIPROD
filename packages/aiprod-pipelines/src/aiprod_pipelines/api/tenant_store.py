"""
AIPROD Multi-Tenant Store
==========================

PostgreSQL-backed persistence layer for multi-tenant SaaS:
- Tenant CRUD with metadata (plan, quotas, contact)
- K8s namespace isolation mapping
- Storage quota tracking & enforcement
- Priority queue weights per tier
- Immutable audit trail for billing & compliance

When ``asyncpg`` is not installed the store operates with an
in-memory dict backend (unit-test / local-dev mode).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------------


class TenantTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class TenantRecord:
    """Persistent tenant record."""

    tenant_id: str = ""
    name: str = ""
    email: str = ""
    tier: TenantTier = TenantTier.FREE
    stripe_customer_id: str = ""
    k8s_namespace: str = ""
    priority_weight: int = 1  # queue priority weight
    storage_quota_gb: float = 10.0
    storage_used_gb: float = 0.0
    max_concurrent_jobs: int = 2
    created_at: float = 0.0
    updated_at: float = 0.0
    active: bool = True

    def __post_init__(self):
        if not self.tenant_id:
            self.tenant_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = time.time()
        self.updated_at = time.time()
        if not self.k8s_namespace:
            self.k8s_namespace = f"tenant-{self.tenant_id[:8]}"

    def storage_ratio(self) -> float:
        return self.storage_used_gb / self.storage_quota_gb if self.storage_quota_gb > 0 else 0.0


# Priority weights by tier
TIER_DEFAULTS: Dict[TenantTier, Dict[str, Any]] = {
    TenantTier.FREE: {
        "priority_weight": 1,
        "storage_quota_gb": 5.0,
        "max_concurrent_jobs": 1,
    },
    TenantTier.PRO: {
        "priority_weight": 5,
        "storage_quota_gb": 100.0,
        "max_concurrent_jobs": 5,
    },
    TenantTier.ENTERPRISE: {
        "priority_weight": 20,
        "storage_quota_gb": 1000.0,
        "max_concurrent_jobs": 50,
    },
}


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------


@dataclass
class AuditEntry:
    """Immutable audit log entry."""

    entry_id: str = ""
    tenant_id: str = ""
    action: str = ""  # created, updated, deleted, quota_change, tier_change, job_submitted …
    details: Dict[str, Any] = field(default_factory=dict)
    actor: str = "system"
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.entry_id:
            self.entry_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = time.time()


# ---------------------------------------------------------------------------
# Storage backend interface
# ---------------------------------------------------------------------------


class TenantBackend:
    """Abstract storage backend."""

    async def get(self, tenant_id: str) -> Optional[TenantRecord]:
        raise NotImplementedError

    async def list_all(self) -> List[TenantRecord]:
        raise NotImplementedError

    async def upsert(self, record: TenantRecord) -> None:
        raise NotImplementedError

    async def delete(self, tenant_id: str) -> None:
        raise NotImplementedError

    async def append_audit(self, entry: AuditEntry) -> None:
        raise NotImplementedError

    async def get_audit_trail(self, tenant_id: str, limit: int = 100) -> List[AuditEntry]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# In-memory backend (local dev / tests)
# ---------------------------------------------------------------------------


class InMemoryBackend(TenantBackend):
    """Dict-backed backend for testing."""

    def __init__(self):
        self._tenants: Dict[str, TenantRecord] = {}
        self._audit: List[AuditEntry] = []

    async def get(self, tenant_id: str) -> Optional[TenantRecord]:
        return self._tenants.get(tenant_id)

    async def list_all(self) -> List[TenantRecord]:
        return list(self._tenants.values())

    async def upsert(self, record: TenantRecord) -> None:
        record.updated_at = time.time()
        self._tenants[record.tenant_id] = record

    async def delete(self, tenant_id: str) -> None:
        self._tenants.pop(tenant_id, None)

    async def append_audit(self, entry: AuditEntry) -> None:
        self._audit.append(entry)

    async def get_audit_trail(self, tenant_id: str, limit: int = 100) -> List[AuditEntry]:
        entries = [e for e in self._audit if e.tenant_id == tenant_id]
        return entries[-limit:]


# ---------------------------------------------------------------------------
# PostgreSQL backend
# ---------------------------------------------------------------------------


class PostgresBackend(TenantBackend):
    """
    asyncpg-based PostgreSQL backend.

    Schema auto-created on first use (tenants + audit_log tables).
    Requires: pip install asyncpg
    """

    DDL = """
    CREATE TABLE IF NOT EXISTS tenants (
        tenant_id       TEXT PRIMARY KEY,
        name            TEXT NOT NULL,
        email           TEXT NOT NULL,
        tier            TEXT NOT NULL DEFAULT 'free',
        stripe_customer TEXT DEFAULT '',
        k8s_namespace   TEXT DEFAULT '',
        priority_weight INTEGER DEFAULT 1,
        storage_quota   REAL DEFAULT 10.0,
        storage_used    REAL DEFAULT 0.0,
        max_concurrent  INTEGER DEFAULT 2,
        created_at      DOUBLE PRECISION NOT NULL,
        updated_at      DOUBLE PRECISION NOT NULL,
        active          BOOLEAN DEFAULT TRUE
    );
    CREATE TABLE IF NOT EXISTS audit_log (
        entry_id    TEXT PRIMARY KEY,
        tenant_id   TEXT NOT NULL REFERENCES tenants(tenant_id),
        action      TEXT NOT NULL,
        details     JSONB DEFAULT '{}',
        actor       TEXT DEFAULT 'system',
        timestamp   DOUBLE PRECISION NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_audit_tenant ON audit_log(tenant_id, timestamp DESC);
    """

    def __init__(self, dsn: str = "postgresql://aiprod:aiprod@localhost:5432/aiprod"):
        self._dsn = dsn
        self._pool: Any = None

    async def _ensure_pool(self):
        if self._pool is None:
            import asyncpg  # type: ignore[import-untyped]

            self._pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)
            async with self._pool.acquire() as conn:
                await conn.execute(self.DDL)

    async def get(self, tenant_id: str) -> Optional[TenantRecord]:
        await self._ensure_pool()
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM tenants WHERE tenant_id = $1", tenant_id
            )
        if not row:
            return None
        return self._row_to_record(row)

    async def list_all(self) -> List[TenantRecord]:
        await self._ensure_pool()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM tenants ORDER BY created_at")
        return [self._row_to_record(r) for r in rows]

    async def upsert(self, record: TenantRecord) -> None:
        await self._ensure_pool()
        record.updated_at = time.time()
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO tenants (tenant_id, name, email, tier, stripe_customer,
                    k8s_namespace, priority_weight, storage_quota, storage_used,
                    max_concurrent, created_at, updated_at, active)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
                ON CONFLICT (tenant_id) DO UPDATE SET
                    name=$2, email=$3, tier=$4, stripe_customer=$5,
                    k8s_namespace=$6, priority_weight=$7, storage_quota=$8,
                    storage_used=$9, max_concurrent=$10, updated_at=$12, active=$13
                """,
                record.tenant_id,
                record.name,
                record.email,
                record.tier.value,
                record.stripe_customer_id,
                record.k8s_namespace,
                record.priority_weight,
                record.storage_quota_gb,
                record.storage_used_gb,
                record.max_concurrent_jobs,
                record.created_at,
                record.updated_at,
                record.active,
            )

    async def delete(self, tenant_id: str) -> None:
        await self._ensure_pool()
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE tenants SET active = FALSE, updated_at = $1 WHERE tenant_id = $2",
                time.time(),
                tenant_id,
            )

    async def append_audit(self, entry: AuditEntry) -> None:
        await self._ensure_pool()
        import json

        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO audit_log (entry_id, tenant_id, action, details, actor, timestamp)
                   VALUES ($1,$2,$3,$4::jsonb,$5,$6)""",
                entry.entry_id,
                entry.tenant_id,
                entry.action,
                json.dumps(entry.details),
                entry.actor,
                entry.timestamp,
            )

    async def get_audit_trail(self, tenant_id: str, limit: int = 100) -> List[AuditEntry]:
        await self._ensure_pool()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM audit_log WHERE tenant_id=$1 ORDER BY timestamp DESC LIMIT $2",
                tenant_id,
                limit,
            )
        return [
            AuditEntry(
                entry_id=r["entry_id"],
                tenant_id=r["tenant_id"],
                action=r["action"],
                details=dict(r["details"]) if r["details"] else {},
                actor=r["actor"],
                timestamp=r["timestamp"],
            )
            for r in rows
        ]

    @staticmethod
    def _row_to_record(row) -> TenantRecord:
        return TenantRecord(
            tenant_id=row["tenant_id"],
            name=row["name"],
            email=row["email"],
            tier=TenantTier(row["tier"]),
            stripe_customer_id=row["stripe_customer"],
            k8s_namespace=row["k8s_namespace"],
            priority_weight=row["priority_weight"],
            storage_quota_gb=row["storage_quota"],
            storage_used_gb=row["storage_used"],
            max_concurrent_jobs=row["max_concurrent"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            active=row["active"],
        )


# ---------------------------------------------------------------------------
# Tenant store (high-level service)
# ---------------------------------------------------------------------------


class TenantStore:
    """
    High-level tenant management service.

    Wraps a backend (InMemory or PostgreSQL) and adds:
    - Default tier configuration
    - Namespace isolation
    - Storage quota enforcement
    - Audit trail on every mutation
    - Priority queue weight management
    """

    def __init__(self, backend: Optional[TenantBackend] = None):
        self._backend = backend or InMemoryBackend()

    async def create_tenant(
        self,
        name: str,
        email: str,
        tier: TenantTier = TenantTier.FREE,
        stripe_customer_id: str = "",
    ) -> TenantRecord:
        """Create a new tenant with tier defaults."""
        defaults = TIER_DEFAULTS.get(tier, TIER_DEFAULTS[TenantTier.FREE])
        record = TenantRecord(
            name=name,
            email=email,
            tier=tier,
            stripe_customer_id=stripe_customer_id,
            priority_weight=defaults["priority_weight"],
            storage_quota_gb=defaults["storage_quota_gb"],
            max_concurrent_jobs=defaults["max_concurrent_jobs"],
        )
        await self._backend.upsert(record)
        await self._backend.append_audit(AuditEntry(
            tenant_id=record.tenant_id,
            action="created",
            details={"tier": tier.value, "email": email},
        ))
        return record

    async def get_tenant(self, tenant_id: str) -> Optional[TenantRecord]:
        return await self._backend.get(tenant_id)

    async def list_tenants(self) -> List[TenantRecord]:
        return await self._backend.list_all()

    async def upgrade_tier(self, tenant_id: str, new_tier: TenantTier) -> Optional[TenantRecord]:
        """Upgrade (or downgrade) tenant tier. Applies new defaults."""
        record = await self._backend.get(tenant_id)
        if not record:
            return None
        old_tier = record.tier
        defaults = TIER_DEFAULTS.get(new_tier, TIER_DEFAULTS[TenantTier.FREE])
        record.tier = new_tier
        record.priority_weight = defaults["priority_weight"]
        record.storage_quota_gb = max(record.storage_quota_gb, defaults["storage_quota_gb"])
        record.max_concurrent_jobs = defaults["max_concurrent_jobs"]
        await self._backend.upsert(record)
        await self._backend.append_audit(AuditEntry(
            tenant_id=tenant_id,
            action="tier_change",
            details={"old_tier": old_tier.value, "new_tier": new_tier.value},
        ))
        return record

    async def update_storage(self, tenant_id: str, delta_gb: float) -> bool:
        """Update storage usage. Returns False if quota would be exceeded."""
        record = await self._backend.get(tenant_id)
        if not record:
            return False
        new_used = record.storage_used_gb + delta_gb
        if new_used > record.storage_quota_gb:
            await self._backend.append_audit(AuditEntry(
                tenant_id=tenant_id,
                action="quota_exceeded",
                details={"used_gb": new_used, "quota_gb": record.storage_quota_gb},
            ))
            return False
        record.storage_used_gb = max(0.0, new_used)
        await self._backend.upsert(record)
        return True

    async def deactivate_tenant(self, tenant_id: str) -> None:
        """Soft-delete a tenant."""
        await self._backend.delete(tenant_id)
        await self._backend.append_audit(AuditEntry(
            tenant_id=tenant_id,
            action="deactivated",
        ))

    async def get_audit_trail(self, tenant_id: str, limit: int = 100) -> List[AuditEntry]:
        return await self._backend.get_audit_trail(tenant_id, limit)

    async def get_priority_weight(self, tenant_id: str) -> int:
        """Get queue priority weight for scheduler integration."""
        record = await self._backend.get(tenant_id)
        return record.priority_weight if record else 1
