"""
Stripe Payment Integration — Cloud billing.
=============================================

Handles:
- Customer creation
- Subscription management
- Usage-based metering (Stripe usage records)
- Invoice generation

This module lives in ``aiprod-cloud`` and is re-exported by the
backward-compatible shim at
``aiprod_pipelines.api.billing_service.StripeIntegration``.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

from aiprod_pipelines.api.billing_service import BillingTier


class StripeIntegration:
    """
    Stripe payment integration (cloud).

    Falls back to local/no-op mode if:
    - The ``stripe`` library is not installed, OR
    - No ``api_key`` is provided.
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
            import stripe as stripe_lib  # noqa: PLC0415

            if api_key:
                stripe_lib.api_key = api_key
                self._stripe = stripe_lib
                self._available = True
            else:
                self._stripe = stripe_lib
                self._available = False
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        return self._available

    # -- Customer ----------------------------------------------------------

    def create_customer(self, tenant_id: str, email: str, name: str) -> Dict[str, Any]:
        if not self._available:
            return {"id": f"cus_local_{tenant_id}", "email": email}
        customer = self._stripe.Customer.create(
            email=email,
            name=name,
            metadata={"tenant_id": tenant_id},
        )
        return {"id": customer.id, "email": customer.email}

    # -- Subscription ------------------------------------------------------

    def create_subscription(self, customer_id: str, tier: BillingTier) -> Dict[str, Any]:
        if not self._available:
            return {"id": f"sub_local_{customer_id}", "status": "active"}
        price_id = self.PRICE_IDS.get(tier, self.PRICE_IDS[BillingTier.PRO])
        sub = self._stripe.Subscription.create(
            customer=customer_id,
            items=[{"price": price_id}],
        )
        return {"id": sub.id, "status": sub.status}

    # -- Usage reporting ---------------------------------------------------

    def report_usage(
        self,
        subscription_item_id: str,
        quantity: int,
        timestamp: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not self._available:
            return {"id": f"mbur_local_{uuid.uuid4().hex[:8]}", "quantity": quantity}
        record = self._stripe.SubscriptionItem.create_usage_record(
            subscription_item_id,
            quantity=quantity,
            timestamp=timestamp or int(time.time()),
            action="increment",
        )
        return {"id": record.id, "quantity": record.quantity}

    # -- Checkout ----------------------------------------------------------

    def create_checkout_session(
        self,
        customer_id: str,
        tier: BillingTier,
        success_url: str = "https://app.aiprod.ai/success",
        cancel_url: str = "https://app.aiprod.ai/cancel",
    ) -> Dict[str, Any]:
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
