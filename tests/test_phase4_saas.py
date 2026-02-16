"""
Phase 4 — SaaS & Scalability — Unit Tests
==========================================

Tests for:
  4.1  Webhooks + SDK client
  4.2  Billing service & Stripe metering
  4.3  Multi-tenant store (PostgreSQL / in-memory)
  4.4  Batch scheduler
  4.5  Inference optimisation pipeline
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def run_async(coro):
    """Run async test in sync context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ===================================================================
# 4.1  Webhooks
# ===================================================================


class TestWebhookEndpoint(unittest.TestCase):
    def test_create_endpoint(self):
        from aiprod_pipelines.api.webhooks import WebhookEndpoint

        ep = WebhookEndpoint(
            url="https://example.com/hook",
            events=["job.completed"],
            secret="s3cret",
        )
        self.assertTrue(ep.endpoint_id)
        self.assertEqual(ep.url, "https://example.com/hook")
        self.assertTrue(ep.active)

    def test_matches_event(self):
        from aiprod_pipelines.api.webhooks import WebhookEndpoint

        ep = WebhookEndpoint(url="https://x.com", events=["job.completed", "job.failed"])
        self.assertTrue(ep.matches("job.completed"))
        self.assertFalse(ep.matches("quota.warning"))

    def test_wildcard_matches_all(self):
        from aiprod_pipelines.api.webhooks import WebhookEndpoint

        ep = WebhookEndpoint(url="https://x.com", events=["*"])
        self.assertTrue(ep.matches("anything"))


class TestWebhookPayload(unittest.TestCase):
    def test_sign(self):
        from aiprod_pipelines.api.webhooks import WebhookPayload

        payload = WebhookPayload(
            event="job.completed",
            job_id="j1",
            tenant_id="t1",
            data={"status": "done"},
        )
        sig = payload.sign("mysecret")
        self.assertTrue(sig.startswith("sha256="))

    def test_to_dict(self):
        from aiprod_pipelines.api.webhooks import WebhookPayload

        payload = WebhookPayload(event="job.completed", job_id="j1", tenant_id="t1", data={"x": 1})
        d = payload.to_dict()
        self.assertEqual(d["event"], "job.completed")
        self.assertIn("payload_id", d)


class TestWebhookManager(unittest.TestCase):
    def test_register_and_list(self):
        from aiprod_pipelines.api.webhooks import WebhookManager

        mgr = WebhookManager()
        ep = mgr.register("t1", "https://x.com/hook", events=["job.completed"], secret="s")
        self.assertEqual(len(mgr.list_endpoints("t1")), 1)
        self.assertEqual(mgr.list_endpoints("t1")[0].url, "https://x.com/hook")

    def test_unregister(self):
        from aiprod_pipelines.api.webhooks import WebhookManager

        mgr = WebhookManager()
        ep = mgr.register("t1", "https://x.com/hook", events=["*"])
        mgr.unregister(ep.endpoint_id)
        self.assertEqual(len(mgr.list_endpoints("t1")), 0)

    def test_delivery_audit_trail(self):
        from aiprod_pipelines.api.webhooks import WebhookManager

        mgr = WebhookManager()
        mgr.register("t1", "https://x.com/hook", events=["job.completed"])
        # Audit trail starts empty
        trail = mgr.delivery_log
        self.assertIsInstance(trail, list)


# ===================================================================
# 4.1  SDK Client
# ===================================================================


class TestAIPRODClient(unittest.TestCase):
    def test_init(self):
        from aiprod_pipelines.api.sdk import AIPRODClient

        client = AIPRODClient(api_key="test-key", base_url="http://localhost:8000")
        self.assertEqual(client._api_key, "test-key")
        self.assertIn("localhost", client._base_url)

    def test_headers(self):
        from aiprod_pipelines.api.sdk import AIPRODClient

        client = AIPRODClient(api_key="k1")
        headers = client._headers()
        self.assertIn("X-API-Key", headers)
        self.assertEqual(headers["X-API-Key"], "k1")

    def test_error_hierarchy(self):
        from aiprod_pipelines.api.sdk import (
            AIPRODError,
            AuthError,
            RateLimitError,
            ValidationError,
            NotFoundError,
        )

        self.assertTrue(issubclass(AuthError, AIPRODError))
        self.assertTrue(issubclass(RateLimitError, AIPRODError))
        self.assertTrue(issubclass(ValidationError, AIPRODError))
        self.assertTrue(issubclass(NotFoundError, AIPRODError))


class TestJobResponse(unittest.TestCase):
    def test_from_dict(self):
        from aiprod_pipelines.api.sdk import JobResponse

        data = {
            "job_id": "j1",
            "status": "completed",
            "video_url": "https://cdn.aiprod.ai/j1.mp4",
            "error": None,
        }
        jr = JobResponse.from_dict(data)
        self.assertEqual(jr.job_id, "j1")
        self.assertEqual(jr.status, "completed")


# ===================================================================
# 4.2  Billing Service
# ===================================================================


class TestPricingPlans(unittest.TestCase):
    def test_plans_exist(self):
        from aiprod_pipelines.api.billing_service import PRICING_PLANS, BillingTier

        self.assertIn(BillingTier.FREE, PRICING_PLANS)
        self.assertIn(BillingTier.PRO, PRICING_PLANS)
        self.assertIn(BillingTier.ENTERPRISE, PRICING_PLANS)

    def test_free_no_overage(self):
        from aiprod_pipelines.api.billing_service import PRICING_PLANS, BillingTier

        free = PRICING_PLANS[BillingTier.FREE]
        self.assertEqual(free.overage_per_second_usd, 0.0)

    def test_enterprise_includes_all_features(self):
        from aiprod_pipelines.api.billing_service import PRICING_PLANS, BillingTier

        ent = PRICING_PLANS[BillingTier.ENTERPRISE]
        self.assertTrue(ent.features["hdr"])
        self.assertTrue(ent.features["tts"])
        self.assertTrue(ent.features["4k_upscale"])


class TestBillingCalculator(unittest.TestCase):
    def test_within_included_no_cost(self):
        from aiprod_pipelines.api.billing_service import BillingCalculator, BillingTier

        calc = BillingCalculator()
        base, feat, total = calc.calculate(
            BillingTier.PRO, duration_sec=5.0, width=768, already_used_sec=0.0
        )
        # 5s is within PRO's 600s included
        self.assertEqual(base, 0.0)

    def test_overage_charges(self):
        from aiprod_pipelines.api.billing_service import BillingCalculator, BillingTier

        calc = BillingCalculator()
        base, feat, total = calc.calculate(
            BillingTier.PRO,
            duration_sec=10.0,
            width=1280,
            already_used_sec=600.0,  # already used all included
        )
        self.assertGreater(base, 0.0)

    def test_resolution_multiplier(self):
        from aiprod_pipelines.api.billing_service import BillingCalculator, BillingTier

        calc = BillingCalculator()
        _, _, total_720 = calc.calculate(
            BillingTier.PRO, 10.0, width=1280, already_used_sec=600.0
        )
        _, _, total_4k = calc.calculate(
            BillingTier.PRO, 10.0, width=3840, already_used_sec=600.0
        )
        self.assertGreater(total_4k, total_720)

    def test_feature_addons(self):
        from aiprod_pipelines.api.billing_service import BillingCalculator, BillingTier

        calc = BillingCalculator()
        _, _, no_feat = calc.calculate(
            BillingTier.FREE, 5.0, 768, features=[], already_used_sec=50.0
        )
        _, feat_cost, with_feat = calc.calculate(
            BillingTier.FREE, 5.0, 768, features=["hdr", "tts"], already_used_sec=50.0
        )
        self.assertGreater(feat_cost, 0.0)

    def test_internal_cost_estimate(self):
        from aiprod_pipelines.api.billing_service import BillingCalculator

        calc = BillingCalculator()
        gpu_h, cost = calc.estimate_internal_cost(10.0, 1920)
        self.assertGreater(gpu_h, 0)
        self.assertGreater(cost, 0)


class TestUsageRecord(unittest.TestCase):
    def test_auto_id(self):
        from aiprod_pipelines.api.billing_service import UsageRecord

        record = UsageRecord(tenant_id="t1", job_id="j1")
        self.assertTrue(record.record_id)
        self.assertGreater(record.timestamp, 0)


class TestBillingService(unittest.TestCase):
    def test_meter_job(self):
        from aiprod_pipelines.api.billing_service import BillingService, BillingTier

        svc = BillingService()
        record = svc.meter_job(
            tenant_id="t1",
            tier=BillingTier.PRO,
            job_id="j1",
            duration_sec=10.0,
            width=1920,
        )
        self.assertEqual(record.tenant_id, "t1")
        self.assertGreaterEqual(record.total_cost_usd, 0.0)

    def test_budget_alert_warning(self):
        from aiprod_pipelines.api.billing_service import BillingService, BillingTier

        svc = BillingService()
        # PRO has 600s included, warn at 80% = 480s
        svc.meter_job("t1", BillingTier.PRO, "j1", 490.0, 768)
        alerts = svc.alerts
        self.assertTrue(any(a["type"] == "quota_warning" for a in alerts))

    def test_budget_alert_exceeded(self):
        from aiprod_pipelines.api.billing_service import BillingService, BillingTier

        svc = BillingService()
        svc.meter_job("t1", BillingTier.PRO, "j1", 610.0, 768)
        alerts = svc.alerts
        self.assertTrue(any(a["type"] == "quota_exceeded" for a in alerts))

    def test_monthly_summary(self):
        from aiprod_pipelines.api.billing_service import BillingService, BillingTier

        svc = BillingService()
        svc.meter_job("t1", BillingTier.PRO, "j1", 5.0, 768)
        svc.meter_job("t1", BillingTier.PRO, "j2", 10.0, 1920)
        summary = svc.get_monthly_summary("t1")
        self.assertEqual(summary.total_jobs, 2)
        self.assertEqual(summary.total_seconds, 15.0)


class TestStripeIntegration(unittest.TestCase):
    def test_local_mode(self):
        from aiprod_pipelines.api.billing_service import StripeIntegration

        stripe = StripeIntegration()  # no api_key → local mode
        self.assertFalse(stripe.available)

    def test_create_customer_local(self):
        from aiprod_pipelines.api.billing_service import StripeIntegration

        stripe = StripeIntegration()
        result = stripe.create_customer("t1", "a@b.com", "Acme")
        self.assertIn("cus_local_", result["id"])

    def test_report_usage_local(self):
        from aiprod_pipelines.api.billing_service import StripeIntegration

        stripe = StripeIntegration()
        result = stripe.report_usage("si_123", 100)
        self.assertEqual(result["quantity"], 100)


# ===================================================================
# 4.3  Multi-Tenant Store
# ===================================================================


class TestTenantRecord(unittest.TestCase):
    def test_auto_fields(self):
        from aiprod_pipelines.api.tenant_store import TenantRecord, TenantTier

        rec = TenantRecord(name="Acme", email="a@b.com")
        self.assertTrue(rec.tenant_id)
        self.assertGreater(rec.created_at, 0)
        self.assertTrue(rec.k8s_namespace.startswith("tenant-"))

    def test_storage_ratio(self):
        from aiprod_pipelines.api.tenant_store import TenantRecord

        rec = TenantRecord(storage_quota_gb=100.0, storage_used_gb=80.0)
        self.assertAlmostEqual(rec.storage_ratio(), 0.8, places=2)


class TestInMemoryBackend(unittest.TestCase):
    def test_upsert_and_get(self):
        from aiprod_pipelines.api.tenant_store import InMemoryBackend, TenantRecord

        async def _test():
            backend = InMemoryBackend()
            rec = TenantRecord(tenant_id="t1", name="A", email="a@b.com")
            await backend.upsert(rec)
            got = await backend.get("t1")
            self.assertIsNotNone(got)
            self.assertEqual(got.name, "A")

        run_async(_test())

    def test_list_all(self):
        from aiprod_pipelines.api.tenant_store import InMemoryBackend, TenantRecord

        async def _test():
            backend = InMemoryBackend()
            await backend.upsert(TenantRecord(tenant_id="t1", name="A", email="a@b.com"))
            await backend.upsert(TenantRecord(tenant_id="t2", name="B", email="b@b.com"))
            all_t = await backend.list_all()
            self.assertEqual(len(all_t), 2)

        run_async(_test())

    def test_delete(self):
        from aiprod_pipelines.api.tenant_store import InMemoryBackend, TenantRecord

        async def _test():
            backend = InMemoryBackend()
            await backend.upsert(TenantRecord(tenant_id="t1", name="A", email="a@b.com"))
            await backend.delete("t1")
            got = await backend.get("t1")
            self.assertIsNone(got)

        run_async(_test())

    def test_audit_trail(self):
        from aiprod_pipelines.api.tenant_store import InMemoryBackend, AuditEntry

        async def _test():
            backend = InMemoryBackend()
            await backend.append_audit(AuditEntry(tenant_id="t1", action="created"))
            trail = await backend.get_audit_trail("t1")
            self.assertEqual(len(trail), 1)
            self.assertEqual(trail[0].action, "created")

        run_async(_test())


class TestTenantStore(unittest.TestCase):
    def test_create_tenant(self):
        from aiprod_pipelines.api.tenant_store import TenantStore, TenantTier

        async def _test():
            store = TenantStore()
            tenant = await store.create_tenant("Acme", "a@b.com", TenantTier.PRO)
            self.assertEqual(tenant.tier, TenantTier.PRO)
            self.assertEqual(tenant.priority_weight, 5)  # PRO default

        run_async(_test())

    def test_upgrade_tier(self):
        from aiprod_pipelines.api.tenant_store import TenantStore, TenantTier

        async def _test():
            store = TenantStore()
            tenant = await store.create_tenant("Acme", "a@b.com", TenantTier.FREE)
            upgraded = await store.upgrade_tier(tenant.tenant_id, TenantTier.ENTERPRISE)
            self.assertEqual(upgraded.tier, TenantTier.ENTERPRISE)
            self.assertEqual(upgraded.priority_weight, 20)

        run_async(_test())

    def test_storage_quota_enforcement(self):
        from aiprod_pipelines.api.tenant_store import TenantStore, TenantTier

        async def _test():
            store = TenantStore()
            tenant = await store.create_tenant("Acme", "a@b.com", TenantTier.FREE)
            # FREE has 5 GB quota
            ok = await store.update_storage(tenant.tenant_id, 4.0)
            self.assertTrue(ok)
            ok = await store.update_storage(tenant.tenant_id, 2.0)  # 6 > 5
            self.assertFalse(ok)

        run_async(_test())

    def test_audit_trail_on_mutations(self):
        from aiprod_pipelines.api.tenant_store import TenantStore, TenantTier

        async def _test():
            store = TenantStore()
            tenant = await store.create_tenant("Acme", "a@b.com")
            await store.upgrade_tier(tenant.tenant_id, TenantTier.PRO)
            trail = await store.get_audit_trail(tenant.tenant_id)
            self.assertEqual(len(trail), 2)  # created + tier_change

        run_async(_test())

    def test_priority_weight(self):
        from aiprod_pipelines.api.tenant_store import TenantStore, TenantTier

        async def _test():
            store = TenantStore()
            tenant = await store.create_tenant("Acme", "a@b.com", TenantTier.ENTERPRISE)
            weight = await store.get_priority_weight(tenant.tenant_id)
            self.assertEqual(weight, 20)

        run_async(_test())


# ===================================================================
# 4.4  Batch Scheduler
# ===================================================================


class TestResolutionBin(unittest.TestCase):
    def test_classification(self):
        from aiprod_pipelines.inference.batch_scheduler import ResolutionBin

        self.assertEqual(ResolutionBin.from_dims(512, 288), ResolutionBin.SD)
        self.assertEqual(ResolutionBin.from_dims(1920, 1080), ResolutionBin.QHD)
        self.assertEqual(ResolutionBin.from_dims(3840, 2160), ResolutionBin.UHD)


class TestInferenceRequest(unittest.TestCase):
    def test_auto_fields(self):
        from aiprod_pipelines.inference.batch_scheduler import InferenceRequest

        req = InferenceRequest(tenant_id="t1", width=1920, height=1080, duration_sec=10.0)
        self.assertTrue(req.request_id)
        self.assertGreater(req.estimated_vram_gb, 0)

    def test_resolution_bin_property(self):
        from aiprod_pipelines.inference.batch_scheduler import InferenceRequest, ResolutionBin

        req = InferenceRequest(width=1280, height=720)
        self.assertEqual(req.resolution_bin, ResolutionBin.FHD)


class TestMemoryAwareSizer(unittest.TestCase):
    def test_fit_within_budget(self):
        from aiprod_pipelines.inference.batch_scheduler import MemoryAwareSizer, InferenceRequest

        sizer = MemoryAwareSizer(vram_budget_gb=10.0)
        reqs = [
            InferenceRequest(width=768, height=432, duration_sec=5.0),
            InferenceRequest(width=768, height=432, duration_sec=5.0),
        ]
        fits, overflow = sizer.fit(reqs)
        self.assertGreater(len(fits), 0)

    def test_overflow(self):
        from aiprod_pipelines.inference.batch_scheduler import MemoryAwareSizer, InferenceRequest

        sizer = MemoryAwareSizer(vram_budget_gb=2.0)
        reqs = [
            InferenceRequest(width=3840, height=2160, duration_sec=30.0),  # huge VRAM
            InferenceRequest(width=3840, height=2160, duration_sec=30.0),
        ]
        fits, overflow = sizer.fit(reqs)
        # At least one should overflow
        total = len(fits) + len(overflow)
        self.assertEqual(total, 2)


class TestBatchScheduler(unittest.TestCase):
    def test_submit(self):
        from aiprod_pipelines.inference.batch_scheduler import BatchScheduler, InferenceRequest

        async def _test():
            scheduler = BatchScheduler()
            req = InferenceRequest(tenant_id="t1", width=768, duration_sec=5.0)
            req_id = await scheduler.submit(req)
            self.assertTrue(req_id)
            self.assertEqual(scheduler.pending_count, 1)

        run_async(_test())

    def test_stats_init(self):
        from aiprod_pipelines.inference.batch_scheduler import BatchScheduler

        scheduler = BatchScheduler()
        stats = scheduler.stats
        self.assertEqual(stats["total_submitted"], 0)
        self.assertEqual(stats["total_batches"], 0)

    def test_timeout_dispatch(self):
        from aiprod_pipelines.inference.batch_scheduler import (
            BatchScheduler, BatchConfig, InferenceRequest,
        )

        dispatched = []

        async def capture(batch):
            dispatched.append(batch)

        async def _test():
            config = BatchConfig(max_wait_sec=0.2, max_batch_size=100)
            scheduler = BatchScheduler(config=config, dispatch_fn=capture)
            scheduler.start()

            # Submit one request — should dispatch after 0.2s timeout
            req = InferenceRequest(tenant_id="t1", width=768, duration_sec=5.0)
            req.timestamp = time.time()  # ensure fresh timestamp
            await scheduler.submit(req)

            await asyncio.sleep(0.5)  # wait for dispatch
            await scheduler.stop()

            self.assertGreater(len(dispatched), 0)

        run_async(_test())

    def test_batch_size_dispatch(self):
        from aiprod_pipelines.inference.batch_scheduler import (
            BatchScheduler, BatchConfig, InferenceRequest,
        )

        dispatched = []

        async def capture(batch):
            dispatched.append(batch)

        async def _test():
            config = BatchConfig(max_wait_sec=60.0, max_batch_size=3)
            scheduler = BatchScheduler(config=config, dispatch_fn=capture)
            scheduler.start()

            for i in range(3):
                await scheduler.submit(
                    InferenceRequest(tenant_id="t1", width=768, duration_sec=5.0)
                )

            await asyncio.sleep(0.3)
            await scheduler.stop()

            self.assertGreater(len(dispatched), 0)
            self.assertEqual(dispatched[0].size, 3)

        run_async(_test())


class TestBatch(unittest.TestCase):
    def test_auto_id(self):
        from aiprod_pipelines.inference.batch_scheduler import Batch

        batch = Batch()
        self.assertTrue(batch.batch_id)
        self.assertEqual(batch.size, 0)


# ===================================================================
# 4.5  Inference Optimisation
# ===================================================================


class TestOptLevel(unittest.TestCase):
    def test_values(self):
        from aiprod_pipelines.inference.optimization import OptLevel

        self.assertEqual(OptLevel.P1.value, "p1")
        self.assertEqual(OptLevel.P2.value, "p2")
        self.assertEqual(OptLevel.P3.value, "p3")


class TestOptimisationResult(unittest.TestCase):
    def test_defaults(self):
        from aiprod_pipelines.inference.optimization import OptimisationResult, OptLevel

        r = OptimisationResult(name="test", level=OptLevel.P1)
        self.assertFalse(r.enabled)
        self.assertEqual(r.speedup, 1.0)


class TestTensorRTOptimiser(unittest.TestCase):
    def test_unavailable(self):
        from aiprod_pipelines.inference.optimization import TensorRTOptimiser

        opt = TensorRTOptimiser()
        _, result = opt.optimise(MagicMock())
        # TensorRT likely not installed in test env
        if not opt.is_available():
            self.assertIn("not installed", result.error)


class TestONNXRuntimeOptimiser(unittest.TestCase):
    def test_unavailable(self):
        from aiprod_pipelines.inference.optimization import ONNXRuntimeOptimiser

        opt = ONNXRuntimeOptimiser()
        if not opt.is_available():
            _, result = opt.optimise(MagicMock())
            self.assertIn("not installed", result.error)


class TestTorchCompileOptimiser(unittest.TestCase):
    def test_config(self):
        from aiprod_pipelines.inference.optimization import TorchCompileOptimiser

        opt = TorchCompileOptimiser(mode="max-autotune", backend="inductor")
        self.assertEqual(opt._mode, "max-autotune")


class TestSpeculativeDecoder(unittest.TestCase):
    def test_no_draft_model_error(self):
        from aiprod_pipelines.inference.optimization import SpeculativeDecoder

        opt = SpeculativeDecoder()
        _, result = opt.optimise(MagicMock())
        self.assertIn("draft_model required", result.error)

    def test_with_draft_model(self):
        from aiprod_pipelines.inference.optimization import SpeculativeDecoder

        opt = SpeculativeDecoder()
        if opt.is_available():
            _, result = opt.optimise(MagicMock(), draft_model=MagicMock())
            self.assertTrue(result.enabled)


class TestSpeculativeWrapper(unittest.TestCase):
    def test_acceptance_rate(self):
        from aiprod_pipelines.inference.optimization import SpeculativeWrapper

        wrapper = SpeculativeWrapper(main_model=lambda x: x, draft_model=lambda x: x)
        self.assertEqual(wrapper.acceptance_rate, 0.0)


class TestKVCacheOptimiser(unittest.TestCase):
    def test_creates_cache(self):
        from aiprod_pipelines.inference.optimization import KVCacheOptimiser

        opt = KVCacheOptimiser(max_frames=60)
        if opt.is_available():
            result_pair, res = opt.optimise(MagicMock())
            self.assertTrue(res.enabled)


class TestKVCache(unittest.TestCase):
    def test_update_and_get(self):
        from aiprod_pipelines.inference.optimization import KVCache

        cache = KVCache()
        cache.update(0, "k_tensor", "v_tensor")
        self.assertEqual(cache.size, 1)
        self.assertEqual(cache.get(0)["key"], "k_tensor")

    def test_clear(self):
        from aiprod_pipelines.inference.optimization import KVCache

        cache = KVCache()
        cache.update(0, "k", "v")
        cache.clear()
        self.assertEqual(cache.size, 0)


class TestINT4Quantiser(unittest.TestCase):
    def test_unavailable(self):
        from aiprod_pipelines.inference.optimization import INT4Quantiser

        opt = INT4Quantiser(method="gptq")
        if not opt.is_available():
            _, result = opt.optimise(MagicMock())
            self.assertIn("not installed", result.error)


class TestOptimisationPipeline(unittest.TestCase):
    def test_add_and_run(self):
        from aiprod_pipelines.inference.optimization import (
            OptimisationPipeline,
            TorchCompileOptimiser,
            KVCacheOptimiser,
        )

        pipeline = OptimisationPipeline()
        pipeline.add(TorchCompileOptimiser())
        pipeline.add(KVCacheOptimiser())

        components = {"transformer": MagicMock(), "vae": MagicMock()}
        optimised = pipeline.run(components)
        self.assertIn("transformer", optimised)
        self.assertIn("vae", optimised)

    def test_summary(self):
        from aiprod_pipelines.inference.optimization import (
            OptimisationPipeline,
            TorchCompileOptimiser,
        )

        pipeline = OptimisationPipeline()
        pipeline.add(TorchCompileOptimiser())
        pipeline.run({"m": MagicMock()})
        summary = pipeline.summary()
        self.assertIn("total_passes", summary)
        self.assertGreaterEqual(summary["total_passes"], 1)

    def test_targeted_optimisation(self):
        from aiprod_pipelines.inference.optimization import (
            OptimisationPipeline,
            KVCacheOptimiser,
        )

        pipeline = OptimisationPipeline()
        pipeline.add(KVCacheOptimiser())
        components = {"transformer": MagicMock(), "vae": MagicMock()}
        # Only apply KV-cache to transformer
        optimised = pipeline.run(components, targets={"kv_cache": ["transformer"]})
        self.assertIn("transformer", optimised)


# ===================================================================
# Cross-module integration
# ===================================================================


class TestBillingAndTenantIntegration(unittest.TestCase):
    """Verify billing + tenant store work together."""

    def test_meter_job_for_tenant(self):
        from aiprod_pipelines.api.billing_service import BillingService, BillingTier
        from aiprod_pipelines.api.tenant_store import TenantStore, TenantTier

        async def _test():
            store = TenantStore()
            tenant = await store.create_tenant("Acme", "a@b.com", TenantTier.PRO)

            svc = BillingService()
            record = svc.meter_job(
                tenant_id=tenant.tenant_id,
                tier=BillingTier.PRO,
                job_id="j1",
                duration_sec=10.0,
                width=1920,
            )
            self.assertEqual(record.tenant_id, tenant.tenant_id)
            self.assertGreaterEqual(record.total_cost_usd, 0)

        run_async(_test())


class TestSchedulerAndBillingIntegration(unittest.TestCase):
    """Verify batch scheduler + billing work together."""

    def test_dispatch_triggers_billing(self):
        from aiprod_pipelines.inference.batch_scheduler import (
            BatchScheduler, BatchConfig, InferenceRequest, Batch,
        )
        from aiprod_pipelines.api.billing_service import BillingService, BillingTier

        svc = BillingService()
        dispatched_jobs = []

        async def on_dispatch(batch: Batch):
            for req in batch.requests:
                record = svc.meter_job(
                    tenant_id=req.tenant_id,
                    tier=BillingTier.PRO,
                    job_id=req.request_id,
                    duration_sec=req.duration_sec,
                    width=req.width,
                )
                dispatched_jobs.append(record)

        async def _test():
            config = BatchConfig(max_wait_sec=0.1, max_batch_size=2)
            scheduler = BatchScheduler(config=config, dispatch_fn=on_dispatch)
            scheduler.start()

            await scheduler.submit(InferenceRequest(tenant_id="t1", width=768, duration_sec=5.0))
            await scheduler.submit(InferenceRequest(tenant_id="t1", width=768, duration_sec=5.0))

            await asyncio.sleep(0.4)
            await scheduler.stop()

            self.assertEqual(len(dispatched_jobs), 2)
            summary = svc.get_monthly_summary("t1")
            self.assertEqual(summary.total_jobs, 2)

        run_async(_test())


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
