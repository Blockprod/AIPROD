"""
Comprehensive tests for Phase 3.3 - Security Audit
"""

import pytest
from datetime import datetime
from src.infra.security_audit import (
    VulnerabilitySeverity,
    SecurityCategory,
    SecurityVulnerability,
    SecurityAuditor,
    SecurityPolicy
)


class TestVulnerabilitySeverity:
    """Test VulnerabilitySeverity enum"""
    
    def test_severity_values(self):
        """Test all severity values"""
        assert VulnerabilitySeverity.CRITICAL.value == "critical"
        assert VulnerabilitySeverity.HIGH.value == "high"
        assert VulnerabilitySeverity.MEDIUM.value == "medium"
        assert VulnerabilitySeverity.LOW.value == "low"
        assert VulnerabilitySeverity.INFO.value == "info"


class TestSecurityCategory:
    """Test SecurityCategory enum"""
    
    def test_category_values(self):
        """Test all security categories"""
        assert SecurityCategory.INJECTION.value == "injection"
        assert SecurityCategory.BROKEN_AUTH.value == "broken_auth"
        assert SecurityCategory.SENSITIVE_DATA.value == "sensitive_data"
        assert SecurityCategory.BROKEN_ACCESS.value == "broken_access"
        assert SecurityCategory.SECURITY_CONFIG.value == "security_config"
        assert SecurityCategory.XSS.value == "xss"


class TestSecurityVulnerability:
    """Test SecurityVulnerability"""
    
    def test_vulnerability_creation(self):
        """Test creating a vulnerability"""
        vuln = SecurityVulnerability(
            category=SecurityCategory.INJECTION,
            title="SQL Injection in login",
            description="User input not properly sanitized",
            severity=VulnerabilitySeverity.CRITICAL,
            affected_component="authentication",
            remediation="Use parameterized queries"
        )
        assert vuln.category == SecurityCategory.INJECTION
        assert vuln.severity == VulnerabilitySeverity.CRITICAL
    
    def test_vulnerability_timestamp(self):
        """Test vulnerability has timestamp"""
        vuln = SecurityVulnerability(
            category=SecurityCategory.XSS,
            title="Reflected XSS",
            description="User input echoed to page",
            severity=VulnerabilitySeverity.HIGH,
            affected_component="search",
            remediation="HTML encode output"
        )
        assert vuln.found_timestamp is not None
        assert isinstance(vuln.found_timestamp, datetime)


class TestSecurityAuditor:
    """Test SecurityAuditor"""
    
    def test_auditor_initialization(self):
        """Test auditor initialization"""
        auditor = SecurityAuditor()
        assert auditor.vulnerabilities == []
        assert auditor.audit_timestamp is not None
    
    def test_get_owasp_checklist(self):
        """Test getting OWASP checklist"""
        auditor = SecurityAuditor()
        checklist = auditor.get_owasp_checklist()
        assert len(checklist) == 10
        assert SecurityCategory.INJECTION in checklist
        assert SecurityCategory.BROKEN_AUTH in checklist
    
    def test_owasp_checklist_structure(self):
        """Test OWASP checklist has correct structure"""
        auditor = SecurityAuditor()
        checklist = auditor.get_owasp_checklist()
        
        for category, details in checklist.items():
            assert "name" in details
            assert "checks" in details
            assert "status" in details
            assert isinstance(details["checks"], list)
    
    def test_check_authentication(self):
        """Test authentication check"""
        auditor = SecurityAuditor()
        auth_check = auditor.check_authentication()
        
        assert auth_check["category"] == "Authentication"
        assert auth_check["overall"] == "PASS"
        assert "jwt_implementation" in auth_check["checks"]
        assert "token_expiration" in auth_check["checks"]
        assert "password_hashing" in auth_check["checks"]
    
    def test_check_authorization(self):
        """Test authorization check"""
        auditor = SecurityAuditor()
        authz_check = auditor.check_authorization()
        
        assert authz_check["category"] == "Authorization"
        assert authz_check["overall"] == "PASS"
        assert "rbac_implemented" in authz_check["checks"]
        assert "permission_checking" in authz_check["checks"]
    
    def test_check_data_protection(self):
        """Test data protection check"""
        auditor = SecurityAuditor()
        dp_check = auditor.check_data_protection()
        
        assert dp_check["category"] == "Data Protection"
        assert dp_check["overall"] == "PASS"
        assert "encryption_at_rest" in dp_check["checks"]
        assert "encryption_in_transit" in dp_check["checks"]
        assert "tls_enforcement" in dp_check["checks"]
    
    def test_check_input_validation(self):
        """Test input validation check"""
        auditor = SecurityAuditor()
        iv_check = auditor.check_input_validation()
        
        assert iv_check["category"] == "Input Validation"
        assert iv_check["overall"] == "PASS"
        assert "sql_injection_protection" in iv_check["checks"]
        assert "xss_prevention" in iv_check["checks"]
        assert "command_injection_protection" in iv_check["checks"]
    
    def test_check_security_headers(self):
        """Test security headers check"""
        auditor = SecurityAuditor()
        sh_check = auditor.check_security_headers()
        
        assert sh_check["category"] == "Security Headers"
        assert sh_check["overall"] == "PASS"
        assert "content_security_policy" in sh_check["headers"]
        assert "x_frame_options" in sh_check["headers"]
        assert "strict_transport_security" in sh_check["headers"]
    
    def test_check_logging_monitoring(self):
        """Test logging and monitoring check"""
        auditor = SecurityAuditor()
        lm_check = auditor.check_logging_monitoring()
        
        assert lm_check["category"] == "Logging & Monitoring"
        assert lm_check["overall"] == "PASS"
        assert "authentication_logging" in lm_check["checks"]
        assert "suspicious_activity_alerting" in lm_check["checks"]
        assert "audit_trail" in lm_check["checks"]
    
    def test_perform_full_audit(self):
        """Test performing full security audit"""
        auditor = SecurityAuditor()
        results = auditor.perform_full_audit()
        
        assert "audit_timestamp" in results
        assert "authentication" in results
        assert "authorization" in results
        assert "data_protection" in results
        assert "input_validation" in results
        assert "security_headers" in results
        assert "logging_monitoring" in results
        assert "vulnerabilities_found" in results
        assert "overall_status" in results
    
    def test_full_audit_with_no_vulnerabilities(self):
        """Test full audit shows PASS when no vulnerabilities"""
        auditor = SecurityAuditor()
        results = auditor.perform_full_audit()
        
        assert results["vulnerabilities_found"] == 0
        assert results["overall_status"] == "PASS"
    
    def test_generate_security_report(self):
        """Test generating security report"""
        auditor = SecurityAuditor()
        report = auditor.generate_security_report()
        
        assert "SECURITY AUDIT REPORT" in report
        assert "Executive Summary" in report
        assert "Category Results" in report
        assert "OWASP Top 10" in report
        assert "Recommendations" in report
    
    def test_report_includes_timestamp(self):
        """Test report includes audit timestamp"""
        auditor = SecurityAuditor()
        report = auditor.generate_security_report()
        
        assert "Generated:" in report
    
    def test_report_includes_authentication_result(self):
        """Test report includes authentication check"""
        auditor = SecurityAuditor()
        report = auditor.generate_security_report()
        
        assert "Authentication:" in report
        assert "PASS" in report
    
    def test_report_includes_authorization_result(self):
        """Test report includes authorization check"""
        auditor = SecurityAuditor()
        report = auditor.generate_security_report()
        
        assert "Authorization:" in report
    
    def test_report_includes_data_protection_result(self):
        """Test report includes data protection check"""
        auditor = SecurityAuditor()
        report = auditor.generate_security_report()
        
        assert "Data Protection:" in report
    
    def test_report_owasp_compliant(self):
        """Test report indicates OWASP compliance"""
        auditor = SecurityAuditor()
        report = auditor.generate_security_report()
        
        assert "OWASP" in report
        assert "compliant" in report.lower()
    
    def test_multiple_auditors_independent(self):
        """Test multiple auditors are independent"""
        auditor1 = SecurityAuditor()
        auditor2 = SecurityAuditor()
        
        vuln = SecurityVulnerability(
            category=SecurityCategory.INJECTION,
            title="Test",
            description="Test",
            severity=VulnerabilitySeverity.HIGH,
            affected_component="test",
            remediation="test"
        )
        auditor1.vulnerabilities.append(vuln)
        
        assert len(auditor1.vulnerabilities) == 1
        assert len(auditor2.vulnerabilities) == 0


class TestSecurityPolicy:
    """Test SecurityPolicy"""
    
    def test_get_password_policy(self):
        """Test getting password policy"""
        policy = SecurityPolicy.get_password_policy()
        
        assert policy["min_length"] == 12
        assert policy["requires_uppercase"] is True
        assert policy["requires_numbers"] is True
        assert policy["requires_special_chars"] is True
        assert policy["expiration_days"] == 90
    
    def test_get_session_policy(self):
        """Test getting session policy"""
        policy = SecurityPolicy.get_session_policy()
        
        assert policy["session_timeout_minutes"] == 30
        assert policy["max_sessions_per_user"] == 5
        assert policy["remember_me_enabled"] is False
    
    def test_get_api_security_policy(self):
        """Test getting API security policy"""
        policy = SecurityPolicy.get_api_security_policy()
        
        assert policy["require_authentication"] is True
        assert policy["rate_limiting_enabled"] is True
        assert policy["request_size_limit_mb"] == 10
    
    def test_get_encryption_policy(self):
        """Test getting encryption policy standards"""
        policies = SecurityPolicy.get_all_policies()
        enc_policy = policies["encryption_policy"]
        
        assert enc_policy["algorithm"] == "AES-256"
        assert enc_policy["tls_version_minimum"] == "1.2"
        assert enc_policy["certificate_renewal_days"] == 30
    
    def test_get_all_policies(self):
        """Test getting all policies"""
        all_policies = SecurityPolicy.get_all_policies()
        
        assert "password_policy" in all_policies
        assert "session_policy" in all_policies
        assert "api_security_policy" in all_policies
        assert "encryption_policy" in all_policies
    
    def test_password_policy_strength(self):
        """Test password policy is strong"""
        policy = SecurityPolicy.get_password_policy()
        
        # Verify strong requirements
        assert policy["min_length"] >= 12
        assert policy["requires_uppercase"] is True
        assert policy["requires_numbers"] is True
        assert policy["requires_special_chars"] is True
    
    def test_session_policy_security(self):
        """Test session policy provides security"""
        policy = SecurityPolicy.get_session_policy()
        
        # Verify session security
        assert policy["session_timeout_minutes"] <= 30
        assert policy["max_sessions_per_user"] > 0
        assert policy["remember_me_enabled"] is False
    
    def test_api_policy_protection(self):
        """Test API policy has protection measures"""
        policy = SecurityPolicy.get_api_security_policy()
        
        assert policy["require_authentication"] is True
        assert policy["rate_limiting_enabled"] is True
        assert policy["request_size_limit_mb"] > 0


class TestSecurityAuditIntegration:
    """Integration tests for security audit"""
    
    def test_full_audit_workflow(self):
        """Test complete audit workflow"""
        auditor = SecurityAuditor()
        
        # Get checklist
        checklist = auditor.get_owasp_checklist()
        assert len(checklist) > 0
        
        # Perform checks
        auth = auditor.check_authentication()
        assert auth["overall"] == "PASS"
        
        authz = auditor.check_authorization()
        assert authz["overall"] == "PASS"
        
        # Generate report
        report = auditor.generate_security_report()
        assert len(report) > 0
    
    def test_comprehensive_security_assessment(self):
        """Test comprehensive security assessment"""
        auditor = SecurityAuditor()
        
        # Run all checks
        checks = [
            auditor.check_authentication(),
            auditor.check_authorization(),
            auditor.check_data_protection(),
            auditor.check_input_validation(),
            auditor.check_security_headers(),
            auditor.check_logging_monitoring()
        ]
        
        # All should pass
        for check in checks:
            assert check["overall"] == "PASS"
    
    def test_policies_support_audit(self):
        """Test security policies support audit requirements"""
        auditor = SecurityAuditor()
        policies = SecurityPolicy.get_all_policies()
        
        # Policies should exist
        assert len(policies) >= 4
        
        # Each policy should have meaningful values
        for policy_name, policy_details in policies.items():
            assert len(policy_details) > 0
    
    def test_audit_report_actionable(self):
        """Test audit report provides actionable recommendations"""
        auditor = SecurityAuditor()
        report = auditor.generate_security_report()
        
        # Report should have recommendations section
        assert "Recommendations" in report
        assert "Conclusion" in report


class TestOWASPCompliance:
    """Test OWASP Top 10 compliance"""
    
    def test_injection_protection(self):
        """Test injection attack protection"""
        auditor = SecurityAuditor()
        checks = auditor.get_owasp_checklist()
        assert SecurityCategory.INJECTION in checks
    
    def test_authentication_security(self):
        """Test authentication security"""
        auditor = SecurityAuditor()
        checks = auditor.get_owasp_checklist()
        assert SecurityCategory.BROKEN_AUTH in checks
    
    def test_sensitive_data_protection(self):
        """Test sensitive data protection"""
        auditor = SecurityAuditor()
        checks = auditor.get_owasp_checklist()
        assert SecurityCategory.SENSITIVE_DATA in checks
    
    def test_access_control(self):
        """Test access control"""
        auditor = SecurityAuditor()
        checks = auditor.get_owasp_checklist()
        assert SecurityCategory.BROKEN_ACCESS in checks
    
    def test_security_configuration(self):
        """Test security misconfiguration prevention"""
        auditor = SecurityAuditor()
        checks = auditor.get_owasp_checklist()
        assert SecurityCategory.SECURITY_CONFIG in checks
    
    def test_xss_prevention(self):
        """Test XSS prevention"""
        auditor = SecurityAuditor()
        checks = auditor.get_owasp_checklist()
        assert SecurityCategory.XSS in checks
    
    def test_deserialization_security(self):
        """Test insecure deserialization prevention"""
        auditor = SecurityAuditor()
        checks = auditor.get_owasp_checklist()
        assert SecurityCategory.INSECURE_DESERIALIZATION in checks
    
    def test_known_vulnerabilities(self):
        """Test known vulnerabilities check"""
        auditor = SecurityAuditor()
        checks = auditor.get_owasp_checklist()
        assert SecurityCategory.COMPONENTS in checks
    
    def test_logging_monitoring(self):
        """Test logging and monitoring"""
        auditor = SecurityAuditor()
        checks = auditor.get_owasp_checklist()
        assert SecurityCategory.LOGGING in checks


class TestSecurityAuditOutput:
    """Test security audit output formats"""
    
    def test_report_format_markdown(self):
        """Test report is in readable format"""
        auditor = SecurityAuditor()
        report = auditor.generate_security_report()
        
        # Should have proper structure
        lines = report.split('\n')
        assert len(lines) > 5
    
    def test_audit_result_status_field(self):
        """Test audit results have status field"""
        auditor = SecurityAuditor()
        results = auditor.perform_full_audit()
        
        assert "overall_status" in results
        assert results["overall_status"] in ["PASS", "REVIEW_NEEDED"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
