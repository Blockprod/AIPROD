"""
Billing and Pricing for Multi-Tenant SaaS.

Handles pricing models, bill generation, invoice management,
and payment tracking.

Core Classes:
  - PricingModel: Pricing configuration
  - BillingCalculator: Bill computation
  - Invoice: Generated invoice
  - SubscriptionPlan: Plan definition
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import threading


class BillingCycle(str, Enum):
    """Billing cycle frequency."""
    MONTHLY = "monthly"
    ANNUAL = "annual"
    QUARTERLY = "quarterly"


class InvoiceStatus(str, Enum):
    """Invoice status."""
    DRAFT = "draft"
    SENT = "sent"
    PAID = "paid"
    OVERDUE = "overdue"
    CANCELLED = "cancelled"


@dataclass
class PricingModel:
    """Pricing configuration."""
    resource_name: str
    base_price_per_unit: float  # Base cost per unit
    unit_name: str = "unit"
    minimum_charge: float = 0.0
    maximum_charge: Optional[float] = None
    volume_discounts: Dict[int, float] = field(default_factory=dict)  # usage -> discount %
    
    def calculate_price(self, units_used: float) -> float:
        """Calculate price for usage amount."""
        if units_used == 0:
            return 0.0
        
        base_cost = units_used * self.base_price_per_unit
        
        # Apply volume discount if applicable
        discount_percentage = 0.0
        for threshold in sorted(self.volume_discounts.keys(), reverse=True):
            if units_used >= threshold:
                discount_percentage = self.volume_discounts[threshold]
                break
        
        discounted_cost = base_cost * (1 - discount_percentage / 100)
        
        # Apply minimum and maximum charges
        final_cost = max(discounted_cost, self.minimum_charge)
        if self.maximum_charge:
            final_cost = min(final_cost, self.maximum_charge)
        
        return final_cost


@dataclass
class SubscriptionPlan:
    """Subscription plan definition."""
    plan_id: str
    name: str
    billing_cycle: BillingCycle
    price: float
    features: Dict[str, Any] = field(default_factory=dict)  # Feature name -> value
    quotas: Dict[str, float] = field(default_factory=dict)  # Quota name -> value
    support_level: str = "community"
    description: str = ""
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plan_id": self.plan_id,
            "name": self.name,
            "billing_cycle": self.billing_cycle.value,
            "price": self.price,
            "support_level": self.support_level,
            "features": self.features,
            "quotas": self.quotas,
        }


@dataclass
class LineItem:
    """Single line item in invoice."""
    item_id: str
    description: str
    quantity: float
    unit_price: float
    amount: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "item_id": self.item_id,
            "description": self.description,
            "quantity": self.quantity,
            "unit_price": self.unit_price,
            "amount": self.amount,
        }


@dataclass
class Invoice:
    """Generated invoice for a tenant."""
    invoice_id: str
    tenant_id: str
    invoice_number: str
    status: InvoiceStatus = InvoiceStatus.DRAFT
    
    issue_date: datetime = field(default_factory=datetime.utcnow)
    due_date: Optional[datetime] = None
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    
    line_items: List[LineItem] = field(default_factory=list)
    subtotal: float = 0.0
    tax_amount: float = 0.0
    tax_rate: float = 0.0
    discount_amount: float = 0.0
    total_amount: float = 0.0
    
    payment_method: Optional[str] = None
    paid_date: Optional[datetime] = None
    paid_amount: float = 0.0
    
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_line_item(self, item: LineItem) -> None:
        """Add line item to invoice."""
        self.line_items.append(item)
    
    def calculate_totals(self) -> None:
        """Calculate invoice totals."""
        self.subtotal = sum(item.amount for item in self.line_items)
        self.tax_amount = self.subtotal * self.tax_rate
        self.total_amount = self.subtotal + self.tax_amount - self.discount_amount
    
    def mark_sent(self) -> None:
        """Mark invoice as sent."""
        self.status = InvoiceStatus.SENT
    
    def mark_paid(self, payment_date: Optional[datetime] = None) -> None:
        """Mark invoice as paid."""
        self.status = InvoiceStatus.PAID
        self.paid_date = payment_date or datetime.utcnow()
        self.paid_amount = self.total_amount
    
    def is_overdue(self) -> bool:
        """Check if invoice is overdue."""
        if self.paid_date or not self.due_date:
            return False
        return datetime.utcnow() > self.due_date
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "invoice_id": self.invoice_id,
            "invoice_number": self.invoice_number,
            "tenant_id": self.tenant_id,
            "status": self.status.value,
            "issue_date": self.issue_date.isoformat(),
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "subtotal": self.subtotal,
            "tax_amount": self.tax_amount,
            "discount_amount": self.discount_amount,
            "total_amount": self.total_amount,
            "line_items_count": len(self.line_items),
            "paid_date": self.paid_date.isoformat() if self.paid_date else None,
        }


class BillingCalculator:
    """Calculates bills and generates invoices."""
    
    def __init__(self):
        """Initialize billing calculator."""
        self._pricing_models: Dict[str, PricingModel] = {}
        self._subscriptions: Dict[str, SubscriptionPlan] = {}
        self._invoices: Dict[str, Invoice] = {}
        self._lock = threading.RLock()
        self._next_invoice_number = 1000
    
    def register_pricing_model(self, model: PricingModel) -> None:
        """Register a pricing model."""
        with self._lock:
            self._pricing_models[model.resource_name] = model
    
    def register_subscription_plan(self, plan: SubscriptionPlan) -> None:
        """Register a subscription plan."""
        with self._lock:
            self._subscriptions[plan.plan_id] = plan
    
    def get_subscription_plan(self, plan_id: str) -> Optional[SubscriptionPlan]:
        """Get subscription plan."""
        with self._lock:
            return self._subscriptions.get(plan_id)
    
    def calculate_usage_charge(self, resource: str, amount: float) -> Tuple[float, str]:
        """Calculate charge for resource usage."""
        with self._lock:
            if resource not in self._pricing_models:
                return 0.0, f"Unknown resource: {resource}"
            
            model = self._pricing_models[resource]
            charge = model.calculate_price(amount)
            return charge, f"Charge calculated for {amount} {model.unit_name}"
    
    def generate_invoice(
        self,
        tenant_id: str,
        period_start: datetime,
        period_end: datetime,
        usage_charges: Dict[str, float],
        plan_charge: float,
        tax_rate: float = 0.0,
        discount_amount: float = 0.0,
    ) -> Invoice:
        """Generate invoice for tenant."""
        with self._lock:
            invoice_number = f"INV-{self._next_invoice_number}"
            self._next_invoice_number += 1
        
        invoice = Invoice(
            invoice_id=f"inv_{tenant_id}_{int(period_start.timestamp())}",
            tenant_id=tenant_id,
            invoice_number=invoice_number,
            period_start=period_start,
            period_end=period_end,
            due_date=period_end + timedelta(days=30),
            tax_rate=tax_rate,
            discount_amount=discount_amount,
        )
        
        # Add plan charge
        if plan_charge > 0:
            plan_item = LineItem(
                item_id="plan",
                description="Subscription Plan",
                quantity=1,
                unit_price=plan_charge,
                amount=plan_charge,
            )
            invoice.add_line_item(plan_item)
        
        # Add usage charges
        for resource, charge in usage_charges.items():
            if charge > 0:
                item = LineItem(
                    item_id=f"usage_{resource}",
                    description=f"Usage: {resource}",
                    quantity=1,
                    unit_price=charge,
                    amount=charge,
                )
                invoice.add_line_item(item)
        
        invoice.calculate_totals()
        
        with self._lock:
            self._invoices[invoice.invoice_id] = invoice
        
        return invoice
    
    def get_invoice(self, invoice_id: str) -> Optional[Invoice]:
        """Get invoice by ID."""
        with self._lock:
            return self._invoices.get(invoice_id)
    
    def get_tenant_invoices(self, tenant_id: str) -> List[Invoice]:
        """Get all invoices for a tenant."""
        with self._lock:
            return [inv for inv in self._invoices.values() if inv.tenant_id == tenant_id]
    
    def get_outstanding_invoices(self, tenant_id: str) -> List[Invoice]:
        """Get unpaid invoices for a tenant."""
        unpaid_statuses = {InvoiceStatus.DRAFT, InvoiceStatus.SENT, InvoiceStatus.OVERDUE}
        with self._lock:
            return [
                inv for inv in self._invoices.values()
                if inv.tenant_id == tenant_id and inv.status in unpaid_statuses
            ]
    
    def get_total_outstanding(self, tenant_id: str) -> float:
        """Get total outstanding amount for tenant."""
        invoices = self.get_outstanding_invoices(tenant_id)
        return sum(inv.total_amount for inv in invoices)


class BillingPortal:
    """Tenant billing information portal."""
    
    def __init__(self, calculator: BillingCalculator):
        """Initialize billing portal."""
        self.calculator = calculator
    
    def get_billing_info(self, tenant_id: str) -> Dict[str, Any]:
        """Get complete billing information for tenant."""
        invoices = self.calculator.get_tenant_invoices(tenant_id)
        
        return {
            "tenant_id": tenant_id,
            "total_invoices": len(invoices),
            "paid_invoices": len([inv for inv in invoices if inv.status == InvoiceStatus.PAID]),
            "outstanding_amount": self.calculator.get_total_outstanding(tenant_id),
            "lifetime_revenue": sum(inv.total_amount for inv in invoices if inv.status == InvoiceStatus.PAID),
            "recent_invoices": [inv.to_dict() for inv in invoices[-5:]],
        }
    
    def get_invoice_details(self, invoice_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed invoice information."""
        invoice = self.calculator.get_invoice(invoice_id)
        if not invoice:
            return None
        
        invoice_dict = invoice.to_dict()
        invoice_dict["line_items"] = [item.to_dict() for item in invoice.line_items]
        return invoice_dict
