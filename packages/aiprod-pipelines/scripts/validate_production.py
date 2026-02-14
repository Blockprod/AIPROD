"""
Production Deployment Validation
=================================

Validates Cloud Run deployment and production readiness:
- Health endpoints
- Environment configuration
- GCP service connectivity
- Performance benchmarks
- Monitoring & alerting

PHASE 5 implementation - Integration & Launch.
"""

import asyncio
import aiohttp
import time
import json
from typing import Dict, Any, List
from datetime import datetime
import argparse


# ============================================================================
# Configuration
# ============================================================================

DEPLOYMENT_CONFIG = {
    "cloud_run_url": "https://aiprod-merger-__PROJECT_ID__.run.app",
    "health_check_timeout": 10,
    "load_test_requests": 20,
    "load_test_concurrency": 4,
    "performance_targets": {
        "health_check_ms": 500,
        "video_request_s": 30,
        "error_rate_percent": 5.0,
        "success_rate_percent": 95.0
    },
    "gcp_services": {
        "cloud_storage": "gs://aiprod-merger-assets",
        "cloud_logging": "projects/__PROJECT_ID__/logs/aiprod-merger",
        "cloud_monitoring": "projects/__PROJECT_ID__/metricDescriptors"
    }
}


# ============================================================================
# Health Check Validation
# ============================================================================

async def check_health_endpoint(base_url: str) -> Dict[str, Any]:
    """
    Check /health endpoint availability and response time.
    
    Returns:
        {
            "passed": bool,
            "response_time_ms": float,
            "status_code": int,
            "details": str
        }
    """
    print("\n[1/6] Health Check Validation")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{base_url}/health",
                timeout=aiohttp.ClientTimeout(total=DEPLOYMENT_CONFIG["health_check_timeout"])
            ) as response:
                response_time_ms = (time.time() - start_time) * 1000
                
                result = {
                    "passed": response.status == 200,
                    "response_time_ms": response_time_ms,
                    "status_code": response.status,
                    "details": ""
                }
                
                if result["passed"]:
                    if response_time_ms < DEPLOYMENT_CONFIG["performance_targets"]["health_check_ms"]:
                        result["details"] = f"✅ Health check passed ({response_time_ms:.0f}ms)"
                        print(result["details"])
                    else:
                        result["details"] = f"⚠️  Health check slow ({response_time_ms:.0f}ms > {DEPLOYMENT_CONFIG['performance_targets']['health_check_ms']}ms)"
                        print(result["details"])
                else:
                    result["details"] = f"❌ Health check failed (HTTP {response.status})"
                    print(result["details"])
                
                return result
                
    except asyncio.TimeoutError:
        return {
            "passed": False,
            "response_time_ms": DEPLOYMENT_CONFIG["health_check_timeout"] * 1000,
            "status_code": 0,
            "details": "❌ Health check timeout"
        }
    except Exception as e:
        return {
            "passed": False,
            "response_time_ms": 0,
            "status_code": 0,
            "details": f"❌ Health check error: {str(e)}"
        }


# ============================================================================
# Environment Configuration Validation
# ============================================================================

async def validate_environment_config(base_url: str) -> Dict[str, Any]:
    """
    Validate environment variables and configuration.
    
    Checks:
        - GCP_PROJECT_ID set
        - BUCKET_NAME configured
        - LOG_LEVEL appropriate
        - API keys present (if required)
    
    Returns:
        {
            "passed": bool,
            "checks": Dict[str, bool],
            "details": str
        }
    """
    print("\n[2/6] Environment Configuration Validation")
    print("-" * 60)
    
    checks = {
        "gcp_project_id": False,
        "bucket_name": False,
        "log_level": False,
        "port": False
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/config") as response:
                if response.status == 200:
                    config = await response.json()
                    
                    checks["gcp_project_id"] = bool(config.get("gcp_project_id"))
                    checks["bucket_name"] = bool(config.get("bucket_name"))
                    checks["log_level"] = config.get("log_level") in ["INFO", "WARNING", "ERROR"]
                    checks["port"] = config.get("port") == 8080
                    
                    passed = all(checks.values())
                    
                    for check, status in checks.items():
                        symbol = "✅" if status else "❌"
                        print(f"  {symbol} {check}: {status}")
                    
                    return {
                        "passed": passed,
                        "checks": checks,
                        "details": "All config checks passed" if passed else "Some config checks failed"
                    }
                else:
                    return {
                        "passed": False,
                        "checks": checks,
                        "details": f"❌ Config endpoint returned HTTP {response.status}"
                    }
                    
    except Exception as e:
        return {
            "passed": False,
            "checks": checks,
            "details": f"❌ Config validation error: {str(e)}"
        }


# ============================================================================
# GCP Services Connectivity
# ============================================================================

async def validate_gcp_connectivity(base_url: str) -> Dict[str, Any]:
    """
    Validate connectivity to GCP services.
    
    Tests:
        - Cloud Storage bucket access
        - Cloud Logging write ability
        - Cloud Monitoring metric submission
    
    Returns:
        {
            "passed": bool,
            "services": Dict[str, bool],
            "details": str
        }
    """
    print("\n[3/6] GCP Services Connectivity")
    print("-" * 60)
    
    services = {
        "cloud_storage": False,
        "cloud_logging": False,
        "cloud_monitoring": False
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            # Test Cloud Storage
            async with session.get(f"{base_url}/gcp/storage/health") as response:
                services["cloud_storage"] = response.status == 200
                symbol = "✅" if services["cloud_storage"] else "❌"
                print(f"  {symbol} Cloud Storage: {services['cloud_storage']}")
            
            # Test Cloud Logging
            async with session.get(f"{base_url}/gcp/logging/health") as response:
                services["cloud_logging"] = response.status == 200
                symbol = "✅" if services["cloud_logging"] else "❌"
                print(f"  {symbol} Cloud Logging: {services['cloud_logging']}")
            
            # Test Cloud Monitoring
            async with session.get(f"{base_url}/gcp/monitoring/health") as response:
                services["cloud_monitoring"] = response.status == 200
                symbol = "✅" if services["cloud_monitoring"] else "❌"
                print(f"  {symbol} Cloud Monitoring: {services['cloud_monitoring']}")
            
            passed = all(services.values())
            
            return {
                "passed": passed,
                "services": services,
                "details": "All GCP services accessible" if passed else "Some GCP services unreachable"
            }
            
    except Exception as e:
        return {
            "passed": False,
            "services": services,
            "details": f"❌ GCP connectivity error: {str(e)}"
        }


# ============================================================================
# Load Testing
# ============================================================================

async def run_production_load_test(base_url: str) -> Dict[str, Any]:
    """
    Run load test against production deployment.
    
    Submits concurrent requests and measures:
        - Success rate
        - Error rate
        - Response times (p50, p95, p99)
        - Throughput
    
    Returns:
        {
            "passed": bool,
            "total_requests": int,
            "successful": int,
            "failed": int,
            "success_rate": float,
            "error_rate": float,
            "response_times": Dict[str, float],
            "throughput_rps": float,
            "details": str
        }
    """
    print("\n[4/6] Load Testing")
    print("-" * 60)
    
    num_requests = DEPLOYMENT_CONFIG["load_test_requests"]
    concurrency = DEPLOYMENT_CONFIG["load_test_concurrency"]
    
    print(f"  Requests: {num_requests}")
    print(f"  Concurrency: {concurrency}")
    print(f"  Target: <{DEPLOYMENT_CONFIG['performance_targets']['error_rate_percent']}% error rate\n")
    
    response_times = []
    successful = 0
    failed = 0
    
    start_time = time.time()
    
    async def make_request(session: aiohttp.ClientSession, request_id: int):
        """Make single test request."""
        nonlocal successful, failed
        
        request_start = time.time()
        
        try:
            async with session.post(
                f"{base_url}/api/v1/video/generate",
                json={
                    "user_prompt": f"Load test request {request_id}",
                    "duration_sec": 10,
                    "complexity": 0.3,
                    "budget_usd": 1.0
                },
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                request_time = time.time() - request_start
                response_times.append(request_time)
                
                if response.status in [200, 202]:
                    successful += 1
                    print(f"  ✅ Request {request_id}: {request_time:.2f}s")
                else:
                    failed += 1
                    print(f"  ❌ Request {request_id}: HTTP {response.status}")
                    
        except asyncio.TimeoutError:
            request_time = time.time() - request_start
            failed += 1
            print(f"  ❌ Request {request_id}: Timeout")
        except Exception as e:
            failed += 1
            print(f"  ❌ Request {request_id}: {str(e)}")
    
    # Execute load test with concurrency control
    async with aiohttp.ClientSession() as session:
        # Process in batches for controlled concurrency
        for i in range(0, num_requests, concurrency):
            batch = [
                make_request(session, request_id)
                for request_id in range(i, min(i + concurrency, num_requests))
            ]
            await asyncio.gather(*batch)
    
    total_time = time.time() - start_time
    
    # Calculate metrics
    success_rate = (successful / num_requests) * 100
    error_rate = (failed / num_requests) * 100
    throughput = num_requests / total_time
    
    # Calculate percentiles
    response_times_sorted = sorted(response_times)
    p50 = response_times_sorted[int(len(response_times_sorted) * 0.50)] if response_times_sorted else 0
    p95 = response_times_sorted[int(len(response_times_sorted) * 0.95)] if response_times_sorted else 0
    p99 = response_times_sorted[int(len(response_times_sorted) * 0.99)] if response_times_sorted else 0
    
    print(f"\n  Results:")
    print(f"    Success Rate: {success_rate:.1f}%")
    print(f"    Error Rate: {error_rate:.1f}%")
    print(f"    Throughput: {throughput:.2f} req/s")
    print(f"    Response Times:")
    print(f"      p50: {p50:.2f}s")
    print(f"      p95: {p95:.2f}s")
    print(f"      p99: {p99:.2f}s")
    
    # Validate against targets
    passed = success_rate >= DEPLOYMENT_CONFIG["performance_targets"]["success_rate_percent"]
    
    return {
        "passed": passed,
        "total_requests": num_requests,
        "successful": successful,
        "failed": failed,
        "success_rate": success_rate,
        "error_rate": error_rate,
        "response_times": {
            "p50": p50,
            "p95": p95,
            "p99": p99
        },
        "throughput_rps": throughput,
        "details": f"{'✅' if passed else '❌'} Success rate: {success_rate:.1f}% (target: {DEPLOYMENT_CONFIG['performance_targets']['success_rate_percent']}%)"
    }


# ============================================================================
# Monitoring & Alerting Validation
# ============================================================================

async def validate_monitoring_alerting(base_url: str) -> Dict[str, Any]:
    """
    Validate monitoring dashboards and alert policies.
    
    Checks:
        - Alert policies configured
        - Monitoring metrics being collected
        - Dashboard accessible
    
    Returns:
        {
            "passed": bool,
            "alerts": Dict[str, bool],
            "metrics": Dict[str, bool],
            "details": str
        }
    """
    print("\n[5/6] Monitoring & Alerting Validation")
    print("-" * 60)
    
    alerts = {
        "error_rate": False,
        "cost_overrun": False,
        "latency": False,
        "quality": False,
        "failure_rate": False
    }
    
    metrics = {
        "cost_tracking": False,
        "quality_scores": False,
        "duration_tracking": False
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            # Check alert policies
            async with session.get(f"{base_url}/monitoring/alerts") as response:
                if response.status == 200:
                    alert_data = await response.json()
                    for alert_name in alerts.keys():
                        alerts[alert_name] = alert_name in alert_data.get("configured_alerts", [])
            
            # Check metrics
            async with session.get(f"{base_url}/monitoring/metrics") as response:
                if response.status == 200:
                    metrics_data = await response.json()
                    for metric_name in metrics.keys():
                        metrics[metric_name] = metric_name in metrics_data.get("active_metrics", [])
            
            # Print results
            print("  Alert Policies:")
            for alert, status in alerts.items():
                symbol = "✅" if status else "⚠️ "
                print(f"    {symbol} {alert}: {status}")
            
            print("\n  Custom Metrics:")
            for metric, status in metrics.items():
                symbol = "✅" if status else "⚠️ "
                print(f"    {symbol} {metric}: {status}")
            
            passed = all(alerts.values()) and all(metrics.values())
            
            return {
                "passed": passed,
                "alerts": alerts,
                "metrics": metrics,
                "details": "All monitoring configured" if passed else "Some monitoring missing"
            }
            
    except Exception as e:
        return {
            "passed": False,
            "alerts": alerts,
            "metrics": metrics,
            "details": f"❌ Monitoring validation error: {str(e)}"
        }


# ============================================================================
# Security Validation
# ============================================================================

async def validate_security(base_url: str) -> Dict[str, Any]:
    """
    Validate security configuration.
    
    Checks:
        - HTTPS enabled
        - Authentication required for sensitive endpoints
        - Rate limiting configured
        - CORS properly configured
    
    Returns:
        {
            "passed": bool,
            "checks": Dict[str, bool],
            "details": str
        }
    """
    print("\n[6/6] Security Validation")
    print("-" * 60)
    
    checks = {
        "https_enabled": base_url.startswith("https://"),
        "rate_limiting": False,
        "cors_configured": False,
        "auth_required": False
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            # Check rate limiting
            async with session.get(f"{base_url}/security/rate-limit") as response:
                checks["rate_limiting"] = response.status == 200
            
            # Check CORS
            async with session.options(f"{base_url}/api/v1/video/generate") as response:
                checks["cors_configured"] = "Access-Control-Allow-Origin" in response.headers
            
            # Check auth (should get 401 without credentials)
            async with session.get(f"{base_url}/api/v1/admin/config") as response:
                checks["auth_required"] = response.status == 401
            
            for check, status in checks.items():
                symbol = "✅" if status else "⚠️ "
                print(f"  {symbol} {check}: {status}")
            
            passed = all(checks.values())
            
            return {
                "passed": passed,
                "checks": checks,
                "details": "All security checks passed" if passed else "Some security checks failed"
            }
            
    except Exception as e:
        return {
            "passed": False,
            "checks": checks,
            "details": f"❌ Security validation error: {str(e)}"
        }


# ============================================================================
# Main Validation Orchestrator
# ============================================================================

async def run_production_validation(base_url: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Run complete production validation suite.
    
    Args:
        base_url: Cloud Run service URL
        verbose: Print detailed output
    
    Returns:
        {
            "passed": bool,
            "total_checks": int,
            "passed_checks": int,
            "failed_checks": int,
            "results": Dict[str, Dict],
            "timestamp": str
        }
    """
    print("\n" + "="*60)
    print("PRODUCTION DEPLOYMENT VALIDATION")
    print("="*60)
    print(f"Target: {base_url}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*60)
    
    # Run all validation tests
    results = {
        "health_check": await check_health_endpoint(base_url),
        "environment": await validate_environment_config(base_url),
        "gcp_connectivity": await validate_gcp_connectivity(base_url),
        "load_test": await run_production_load_test(base_url),
        "monitoring": await validate_monitoring_alerting(base_url),
        "security": await validate_security(base_url)
    }
    
    # Calculate overall status
    passed_checks = sum(1 for r in results.values() if r["passed"])
    total_checks = len(results)
    overall_passed = passed_checks == total_checks
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        symbol = "✅" if result["passed"] else "❌"
        print(f"{symbol} {test_name.replace('_', ' ').title()}: {result['details']}")
    
    print("\n" + "="*60)
    print(f"Overall: {passed_checks}/{total_checks} checks passed")
    print(f"Status: {'✅ PRODUCTION READY' if overall_passed else '❌ NOT READY FOR PRODUCTION'}")
    print("="*60 + "\n")
    
    return {
        "passed": overall_passed,
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "failed_checks": total_checks - passed_checks,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Validate AIPROD production deployment")
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="Cloud Run service URL (e.g., https://aiprod-merger-xxx.run.app)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Run validation
    result = asyncio.run(run_production_validation(args.url, args.verbose))
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # Exit with appropriate code
    exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
