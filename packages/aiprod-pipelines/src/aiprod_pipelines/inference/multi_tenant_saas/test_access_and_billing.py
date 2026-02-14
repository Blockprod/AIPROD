"""
Comprehensive tests for access control and billing.

Tests:
  - Permission and Role
  - RBAC engine
  - Permission checker
  - Pricing and billing
  - Invoice generation
  - Billing portal
"""

import unittest
from datetime import datetime, timedelta

from aiprod_pipelines.inference.multi_tenant_saas.access_control import (
    Permission,
    Role,
    UserRole,
    ResourceType,
    Action,
    RoleType,
    RoleBasedAccessControl,
    PermissionChecker,
)

from aiprod_pipelines.inference.multi_tenant_saas.billing import (
    PricingModel,
    SubscriptionPlan,
    LineItem,
    Invoice,
    BillingCalculator,
    BillingPortal,
    BillingCycle,
    InvoiceStatus,
)


class TestPermissionAndRole(unittest.TestCase):
    """Test permissions and roles."""
    
    def test_create_permission(self):
        """Test permission creation."""
        perm = Permission(ResourceType.VIDEO_PROJECT, Action.CREATE)
        
        self.assertEqual(perm.resource, ResourceType.VIDEO_PROJECT)
        self.assertEqual(perm.action, Action.CREATE)
    
    def test_permission_string_conversion(self):
        """Test permission string conversion."""
        perm1 = Permission(ResourceType.MODEL, Action.DELETE)
        perm_string = perm1.to_string()
        
        perm2 = Permission.from_string(perm_string)
        self.assertEqual(perm1, perm2)
    
    def test_create_role(self):
        """Test role creation."""
        role = Role(
            role_id="custom_role",
            name="Custom Role",
            role_type=RoleType.CUSTOM,
        )
        
        self.assertEqual(role.role_id, "custom_role")
        self.assertEqual(role.name, "Custom Role")
    
    def test_add_remove_permissions(self):
        """Test adding/removing permissions to role."""
        role = Role(role_id="role1", name="Role1", role_type=RoleType.CUSTOM)
        
        perm1 = Permission(ResourceType.VIDEO_PROJECT, Action.CREATE)
        perm2 = Permission(ResourceType.MODEL, Action.READ)
        
        role.add_permission(perm1)
        role.add_permission(perm2)
        
        self.assertTrue(role.has_permission(perm1))
        self.assertTrue(role.has_permission(perm2))
        
        role.remove_permission(perm1)
        self.assertFalse(role.has_permission(perm1))


class TestRBAC(unittest.TestCase):
    """Test RBAC engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rbac = RoleBasedAccessControl()
    
    def test_system_roles_exist(self):
        """Test that system roles are created."""
        admin_role = self.rbac.get_role("system_admin")
        self.assertIsNotNone(admin_role)
        self.assertEqual(admin_role.role_type, RoleType.ADMIN)
    
    def test_assign_role_to_user(self):
        """Test assigning role to user."""
        success = self.rbac.assign_role_to_user("t1", "user1", "system_user")
        self.assertTrue(success)
        
        roles = self.rbac.get_user_roles("t1", "user1")
        self.assertEqual(len(roles), 1)
        self.assertEqual(roles[0].role_id, "system_user")
    
    def test_admin_has_all_permissions(self):
        """Test admin has all permissions."""
        self.rbac.assign_role_to_user("t1", "admin_user", "system_admin")
        
        # Admin should have all permissions
        has_perm = self.rbac.check_permission(
            "t1",
            "admin_user",
            ResourceType.BILLING,
            Action.ADMIN,
        )
        self.assertTrue(has_perm)
    
    def test_user_role_permissions(self):
        """Test user role permissions."""
        self.rbac.assign_role_to_user("t1", "user1", "system_user")
        
        # User can create videos
        can_create = self.rbac.check_permission(
            "t1",
            "user1",
            ResourceType.VIDEO_PROJECT,
            Action.CREATE,
        )
        self.assertTrue(can_create)
        
        # User cannot approve billing changes
        can_approve = self.rbac.check_permission(
            "t1",
            "user1",
            ResourceType.BILLING,
            Action.APPROVE,
        )
        self.assertFalse(can_approve)
    
    def test_custom_role(self):
        """Test creating custom role."""
        perms = {
            Permission(ResourceType.VIDEO_PROJECT, Action.CREATE),
            Permission(ResourceType.VIDEO_PROJECT, Action.READ),
        }
        
        custom_role = self.rbac.create_custom_role(
            "t1",
            "Video Creator",
            perms,
        )
        
        self.assertIsNotNone(custom_role)
        self.assertEqual(len(custom_role.permissions), 2)
    
    def test_resource_specific_access(self):
        """Test resource-specific permission."""
        self.rbac.assign_role_to_user("t1", "user1", "system_user")
        
        # Grant specific resource access
        perm = Permission(ResourceType.VIDEO_PROJECT, Action.DELETE)
        self.rbac.grant_resource_access(
            "t1",
            "user1",
            "project_123",
            {perm},
        )
        
        # Check resource-specific access
        can_delete = self.rbac.check_resource_permission(
            "t1",
            "user1",
            "project_123",
            ResourceType.VIDEO_PROJECT,
            Action.DELETE,
        )
        self.assertTrue(can_delete)


class TestPermissionChecker(unittest.TestCase):
    """Test permission checker utility."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rbac = RoleBasedAccessControl()
        self.checker = PermissionChecker(self.rbac)
    
    def test_require_permission(self):
        """Test permission requirement check."""
        self.rbac.assign_role_to_user("t1", "user1", "system_user")
        
        allowed, message = self.checker.require_permission(
            "t1",
            "user1",
            ResourceType.VIDEO_PROJECT,
            Action.CREATE,
        )
        
        self.assertTrue(allowed)


class TestPricingModel(unittest.TestCase):
    """Test pricing models."""
    
    def test_calculate_price_no_discount(self):
        """Test basic price calculation."""
        model = PricingModel(
            resource_name="api_calls",
            base_price_per_unit=0.01,
        )
        
        price = model.calculate_price(100)
        self.assertEqual(price, 1.0)
    
    def test_volume_discount(self):
        """Test volume discount."""
        model = PricingModel(
            resource_name="api_calls",
            base_price_per_unit=1.0,
            volume_discounts={100: 10, 1000: 20},  # 10% and 20% discount
        )
        
        price_100 = model.calculate_price(100)
        price_1000 = model.calculate_price(1000)
        
        self.assertEqual(price_100, 90.0)  # 100 * 1.0 * 0.9
        self.assertEqual(price_1000, 800.0)  # 1000 * 1.0 * 0.8
    
    def test_minimum_charge(self):
        """Test minimum charge."""
        model = PricingModel(
            resource_name="storage",
            base_price_per_unit=0.01,
            minimum_charge=10.0,
        )
        
        # Even small usage should incur minimum
        price = model.calculate_price(50)
        self.assertEqual(price, 10.0)
    
    def test_maximum_charge(self):
        """Test maximum charge cap."""
        model = PricingModel(
            resource_name="compute",
            base_price_per_unit=1.0,
            maximum_charge=100.0,
        )
        
        # Large usage should be capped
        price = model.calculate_price(1000)
        self.assertEqual(price, 100.0)


class TestSubscriptionPlan(unittest.TestCase):
    """Test subscription plans."""
    
    def test_create_plan(self):
        """Test plan creation."""
        plan = SubscriptionPlan(
            plan_id="pro",
            name="Professional",
            billing_cycle=BillingCycle.MONTHLY,
            price=99.99,
        )
        
        self.assertEqual(plan.plan_id, "pro")
        self.assertEqual(plan.price, 99.99)


class TestBillingCalculator(unittest.TestCase):
    """Test billing calculator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calc = BillingCalculator()
        
        # Register pricing model
        model = PricingModel(
            resource_name="video_generation",
            base_price_per_unit=0.50,
        )
        self.calc.register_pricing_model(model)
        
        # Register plan
        plan = SubscriptionPlan(
            plan_id="pro",
            name="Professional",
            billing_cycle=BillingCycle.MONTHLY,
            price=99.99,
        )
        self.calc.register_subscription_plan(plan)
    
    def test_calculate_usage_charge(self):
        """Test usage charge calculation."""
        charge, message = self.calc.calculate_usage_charge("video_generation", 10)
        
        self.assertEqual(charge, 5.0)  # 10 * 0.50
    
    def test_generate_invoice(self):
        """Test invoice generation."""
        start = datetime.utcnow()
        end = start + timedelta(days=30)
        
        invoice = self.calc.generate_invoice(
            tenant_id="t1",
            period_start=start,
            period_end=end,
            usage_charges={"video_generation": 50.0},
            plan_charge=99.99,
            tax_rate=0.1,
        )
        
        self.assertIsNotNone(invoice)
        self.assertEqual(invoice.tenant_id, "t1")
        self.assertEqual(invoice.subtotal, 149.99)  # 99.99 + 50.0
    
    def test_invoice_lifecycle(self):
        """Test invoice lifecycle."""
        invoice = self.calc.generate_invoice(
            tenant_id="t1",
            period_start=datetime.utcnow(),
            period_end=datetime.utcnow() + timedelta(days=30),
            usage_charges={},
            plan_charge=0,
        )
        
        self.assertEqual(invoice.status, InvoiceStatus.DRAFT)
        
        invoice.mark_sent()
        self.assertEqual(invoice.status, InvoiceStatus.SENT)
        
        invoice.mark_paid()
        self.assertEqual(invoice.status, InvoiceStatus.PAID)
        self.assertIsNotNone(invoice.paid_date)
    
    def test_get_outstanding_invoices(self):
        """Test getting outstanding invoices."""
        for i in range(3):
            inv = self.calc.generate_invoice(
                tenant_id="t1",
                period_start=datetime.utcnow() - timedelta(days=i*30),
                period_end=datetime.utcnow() - timedelta(days=(i-1)*30),
                usage_charges={},
                plan_charge=10.0,
            )
            if i == 0:
                inv.mark_paid()
        
        outstanding = self.calc.get_outstanding_invoices("t1")
        self.assertEqual(len(outstanding), 2)


class TestBillingPortal(unittest.TestCase):
    """Test billing portal."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calc = BillingCalculator()
        self.portal = BillingPortal(self.calc)
    
    def test_get_billing_info(self):
        """Test getting billing information."""
        # Generate some invoices
        for i in range(2):
            self.calc.generate_invoice(
                tenant_id="t1",
                period_start=datetime.utcnow() - timedelta(days=30),
                period_end=datetime.utcnow(),
                usage_charges={},
                plan_charge=50.0,
            )
        
        info = self.portal.get_billing_info("t1")
        
        self.assertEqual(info["tenant_id"], "t1")
        self.assertEqual(info["total_invoices"], 2)


if __name__ == "__main__":
    unittest.main()
