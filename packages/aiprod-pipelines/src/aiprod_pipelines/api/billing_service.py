"""
AIPROD Billing Service
=======================

Production billing with Stripe integration:
- Usage-based metering (per-second video, resolution multiplier, feature add-ons)
- Subscription tiers (Free / Pro / Enterprise)
- Stripe Checkout & usage-record reporting
- Budget alerts (quota warning / exceeded)
- Internal cost-per-video tracking (GPU-hours × rate)

Stripe is optional: when the stripe library is not installed, the service
operates in local-only mode with in-memory ledger.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Pricing model
# ---------------------------------------------------------------------------

class BillingTier(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class PricingPlan:
    """Pricing plan definition."""

    tier: BillingTier
    monthly_price_usd: float  # base subscription price
    included_seconds: float  # included video seconds per month
    overage_per_second_usd: float  # per-second overage cost
    max_resolution: int = 1920  # max width
    features: Dict[str, bool] = field(default_factory=dict)

    # Resolution multipliers
    RESOLUTION_MULTIPLIER: Dict[int, float] = field(default_factory=lambda: {
        512: 0.5,
        768: 0.75,
        1280: 1.0,
        1920: 1.5,
        3840: 3.0,
    })

    # Feature add-on costs per second
    FEATURE_COSTS: Dict[str, float] = field(default_factory=lambda: {
        "hdr": 0.005,
        "tts": 0.010,
        "lip_sync": 0.008,
        "4k_upscale": 0.020,
        "custom_lut": 0.003,
    })


# Default plans
PRICING_PLANS: Dict[BillingTier, PricingPlan] = {
    BillingTier.FREE: PricingPlan(
        tier=BillingTier.FREE,
        monthly_price_usd=0.0,
        included_seconds=30.0,  # 30s free per month
        overage_per_second_usd=0.0,  # no overage for free
        max_resolution=768,
        features={"hdr": False, "tts": False, "4k_upscale": False},
    ),
    BillingTier.PRO: PricingPlan(
        tier=BillingTier.PRO,
        monthly_price_usd=49.0,
        included_seconds=600.0,  # 10 min included
        overage_per_second_usd=0.05,
        max_resolution=1920,
        features={"hdr": True, "tts": True, "4k_upscale": False},
    ),
    BillingTier.ENTERPRISE: PricingPlan(
        tier=BillingTier.ENTERPRISE,
        monthly_price_usd=499.0,
        included_seconds=6000.0,  # 100 min included
        overage_per_second_usd=0.03,
        max_resolution=3840,
        features={"hdr": True, "tts": True, "lip_sync": True, "4k_upscale": True, "custom_lut": True},
    ),
}


# ---------------------------------------------------------------------------
# Usage record & metering
# ---------------------------------------------------------------------------


@dataclass
class UsageRecord:
    """A single metered usage event."""

    record_id: str = ""
    tenant_id: str = ""
    job_id: str = ""
    timestamp: float = 0.0
    video_duration_sec: float = 0.0
    width: int = 768
    height: int = 512
    features_used: List[str] = field(default_factory=list)
    base_cost_usd: float = 0.0
    feature_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    gpu_hours: float = 0.0
    internal_cost_usd: float = 0.0  # actual compute cost

    def __post_init__(self):
        if not self.record_id:
            self.record_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = time.time()


@dataclass
class MonthlyUsageSummary:
    """Aggregated monthly usage for a tenant."""

    tenant_id: str
    month: str  # "2026-02"
    tier: BillingTier
    total_seconds: float = 0.0
    total_jobs: int = 0
    base_cost_usd: float = 0.0
    feature_cost_usd: float = 0.0
    overage_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    internal_cost_usd: float = 0.0
    margin_usd: float = 0.0
    records: List[UsageRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Billing calculator
# ---------------------------------------------------------------------------


class BillingCalculator:
    """
    Calculates video generation costs based on pricing plan.

    Factors:
    - Base: duration_sec × overage_rate (or included)
    - Resolution multiplier (512p=0.5×, 1080p=1.5×, 4K=3×)
    - Feature add-ons (HDR, TTS, lip-sync, etc.)
    """

    def __init__(self, plans: Optional[Dict[BillingTier, PricingPlan]] = None):
        self._plans = plans or PRICING_PLANS

    def calculate(
        self,
        tier: BillingTier,
        duration_sec: float,
        width: int,
        features: Optional[List[str]] = None,
        already_used_sec: float = 0.0,
    ) -> Tuple[float, float, float]:
        """
        Calculate cost for a single video generation.

        Args:
            tier: Customer billing tier.
            duration_sec: Video duration in seconds.
            width: Video width (determines resolution multiplier).
            features: List of features used (hdr, tts, etc.).
            already_used_sec: Seconds already used this month.

        Returns:
            (base_cost, feature_cost, total_cost) in USD
        """
        plan = self._plans[tier]
        features = features or []

        # Resolution multiplier
        res_mult = 1.0
        for res_w in sorted(plan.RESOLUTION_MULTIPLIER.keys()):
            if width <= res_w:
                res_mult = plan.RESOLUTION_MULTIPLIER[res_w]
                break
        else:
            res_mult = plan.RESOLUTION_MULTIPLIER.get(3840, 3.0)

        # Base cost (check included seconds)
        remaining_included = max(0, plan.included_seconds - already_used_sec)
        billable_seconds = max(0, duration_sec - remaining_included)
        base_cost = billable_seconds * plan.overage_per_second_usd * res_mult

        # Feature add-on costs
        feature_cost = 0.0
        for feat in features:
            feat_rate = plan.FEATURE_COSTS.get(feat, 0.0)
            if plan.features.get(feat, False):
                # Feature included in plan — no extra cost
                continue
            feature_cost += feat_rate * duration_sec

        total = base_cost + feature_cost
        return round(base_cost, 4), round(feature_cost, 4), round(total, 4)

    def estimate_internal_cost(
        self, duration_sec: float, width: int, gpu_cost_per_hour: float = 2.50
    ) -> Tuple[float, float]:
        """
        Estimate internal compute cost (GPU-hours × rate).

        Returns (gpu_hours, cost_usd).
        """
        # Empirical model: ~2 GPU-minutes per second of video at 1080p
        pixels = width * (width * 9 // 16)  # approximate
        base_gpu_min = duration_sec * 2.0
        res_factor = pixels / (1920 * 1080)
        gpu_minutes = base_gpu_min * max(0.5, res_factor)
        gpu_hours = gpu_minutes / 60
        cost = gpu_hours * gpu_cost_per_hour
        return round(gpu_hours, 4), round(cost, 4)


# ---------------------------------------------------------------------------
# Stripe integration
# ---------------------------------------------------------------------------


class StripeIntegration:
    """
    Stripe payment integration.

    Handles:
    - Customer creation
    - Subscription management
    - Usage-based metering (Stripe usage records)
    - Invoice generation

    Requires: pip install stripe
    Falls back to no-op mode if stripe is not installed.
    """

    PRICE_IDS: Dict[BillingTier, str] = {
        BillingTier.FREE: "price_free_placeholder",
        BillingTier.PRO: "price_pro_placeholder",
        BillingTier.ENTERPRISE: "price_enterprise_placeholder",
    }

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key
        self._stripe = None
        self._available = False
        try:
            import stripe as stripe_lib

            if api_key:
                stripe_lib.api_key = api_key
            self._stripe = stripe_lib
            self._available = True
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        return self._available

    def create_customer(self, tenant_id: str, email: str, name: str) -> Dict[str, Any]:
        """Create a Stripe customer."""
        if not self._available:
            return {"id": f"cus_local_{tenant_id}", "email": email}
        customer = self._stripe.Customer.create(
            email=email,
            name=name,
            metadata={"tenant_id": tenant_id},
        )
        return {"id": customer.id, "email": customer.email}

    def create_subscription(
        self, customer_id: str, tier: BillingTier
    ) -> Dict[str, Any]:
        """Create a subscription for a customer."""
        if not self._available:
            return {"id": f"sub_local_{customer_id}", "status": "active"}
        price_id = self.PRICE_IDS.get(tier, self.PRICE_IDS[BillingTier.PRO])
        sub = self._stripe.Subscription.create(
            customer=customer_id,
            items=[{"price": price_id}],
        )
        return {"id": sub.id, "status": sub.status}

    def report_usage(
        self,
        subscription_item_id: str,
        quantity: int,
        timestamp: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Report metered usage to Stripe."""
        if not self._available:
            return {"id": f"mbur_local_{uuid.uuid4().hex[:8]}", "quantity": quantity}
        record = self._stripe.SubscriptionItem.create_usage_record(
            subscription_item_id,
            quantity=quantity,
            timestamp=timestamp or int(time.time()),
            action="increment",
        )
        return {"id": record.id, "quantity": record.quantity}

    def create_checkout_session(
        self,
        customer_id: str,
        tier: BillingTier,
        success_url: str = "https://app.aiprod.ai/success",
        cancel_url: str = "https://app.aiprod.ai/cancel",
    ) -> Dict[str, Any]:
        """Create a Stripe Checkout session."""
        if not self._available:
            return {"url": f"{success_url}?session=local_{uuid.uuid4().hex[:8]}"}
        price_id = self.PRICE_IDS.get(tier, self.PRICE_IDS[BillingTier.PRO])
        session = self._stripe.checkout.Session.create(
            customer=customer_id,
            payment_method_types=["card"],
            line_items=[{"price": price_id, "quantity": 1}],
            mode="subscription",
            success_url=success_url,
            cancel_url=cancel_url,
        )
        return {"url": session.url, "id": session.id}


# ---------------------------------------------------------------------------
# Billing Service (orchestrator)
# ---------------------------------------------------------------------------


class BillingService:
    """
    Orchestrates metering, cost calculation, Stripe reporting, and budget alerts.

    Usage:
        svc = BillingService()
        record = svc.meter_job(tenant_id="t1", tier=BillingTier.PRO,
                               job_id="j1", duration_sec=10.0, width=1920,
                               features=["tts"])
        summary = svc.get_monthly_summary("t1", "2026-02")
    """

    # Budget alert thresholds (fraction of included seconds)
    QUOTA_WARNING_THRESHOLD = 0.80
    QUOTA_EXCEEDED_THRESHOLD = 1.00

    def __init__(
        self,
        calculator: Optional[BillingCalculator] = None,
        stripe: Optional[StripeIntegration] = None,
    ):
        self._calculator = calculator or BillingCalculator()
        self._stripe = stripe or StripeIntegration()
        self._records: Dict[str, List[UsageRecord]] = {}  # tenant_id → records
        self._alerts: List[Dict[str, Any]] = []

    def meter_job(
        self,
        tenant_id: str,
        tier: BillingTier,
        job_id: str,
        duration_sec: float,
        width: int,
        height: int = 0,
        features: Optional[List[str]] = None,
        gpu_hours: float = 0.0,
    ) -> UsageRecord:
        """
        Record a completed job for billing.

        Calculates costs, records usage, checks budget alerts.
        """
        features = features or []

        # Get already-used seconds this month
        records = self._records.get(tenant_id, [])
        already_used = sum(r.video_duration_sec for r in records)

        # Calculate cost
        base, feat, total = self._calculator.calculate(
            tier, duration_sec, width, features, already_used
        )

        # Internal cost
        if gpu_hours <= 0:
            gpu_h, internal = self._calculator.estimate_internal_cost(duration_sec, width)
        else:
            gpu_h = gpu_hours
            internal = gpu_hours * 2.50

        record = UsageRecord(
            tenant_id=tenant_id,
            job_id=job_id,
            video_duration_sec=duration_sec,
            width=width,
            height=height or (width * 9 // 16),
            features_used=features,
            base_cost_usd=base,
            feature_cost_usd=feat,
            total_cost_usd=total,
            gpu_hours=gpu_h,
            internal_cost_usd=internal,
        )

        self._records.setdefault(tenant_id, []).append(record)

        # Budget alert check
        plan = PRICING_PLANS.get(tier)
        if plan:
            new_total = already_used + duration_sec
            ratio = new_total / plan.included_seconds if plan.included_seconds > 0 else 0
            if ratio >= self.QUOTA_EXCEEDED_THRESHOLD:
                self._alerts.append({
                    "type": "quota_exceeded",
                    "tenant_id": tenant_id,
                    "used_sec": new_total,
                    "included_sec": plan.included_seconds,
                    "timestamp": time.time(),
                })
            elif ratio >= self.QUOTA_WARNING_THRESHOLD:
                self._alerts.append({
                    "type": "quota_warning",
                    "tenant_id": tenant_id,
                    "used_sec": new_total,
                    "included_sec": plan.included_seconds,
                    "timestamp": time.time(),
                })

        return record

    def get_monthly_summary(self, tenant_id: str, month: str = "") -> MonthlyUsageSummary:
        """Get aggregated usage summary for a tenant."""
        records = self._records.get(tenant_id, [])
        tier = BillingTier.FREE  # default; would be looked up from tenant store
        plan = PRICING_PLANS.get(tier)

        total_sec = sum(r.video_duration_sec for r in records)
        base_total = sum(r.base_cost_usd for r in records)
        feat_total = sum(r.feature_cost_usd for r in records)
        internal_total = sum(r.internal_cost_usd for r in records)
        total_cost = base_total + feat_total

        return MonthlyUsageSummary(
            tenant_id=tenant_id,
            month=month or time.strftime("%Y-%m"),
            tier=tier,
            total_seconds=total_sec,
            total_jobs=len(records),
            base_cost_usd=round(base_total, 4),
            feature_cost_usd=round(feat_total, 4),
            overage_cost_usd=round(base_total, 4),  # base is entirely overage
            total_cost_usd=round(total_cost, 4),
            internal_cost_usd=round(internal_total, 4),
            margin_usd=round(total_cost - internal_total, 4),
            records=records,
        )

    @property
    def alerts(self) -> List[Dict[str, Any]]:
        return list(self._alerts)

    def clear_alerts(self) -> None:
        self._alerts.clear()
