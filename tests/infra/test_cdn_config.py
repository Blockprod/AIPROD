"""
Tests for CDN Integration
Validates cache policies and header application
"""

import pytest
from src.infra.cdn_config import CDNConfig, get_cdn_config, CachePolicy


class TestCDNConfiguration:
    """Test CDN configuration"""
    
    def test_cdn_config_initialization(self):
        """Test CDN config initializes correctly"""
        cdn = CDNConfig()
        assert cdn.policies is not None
        assert cdn.cache_directives is not None
        assert cdn.monitoring is not None
    
    def test_cdn_singleton(self):
        """Test CDN config is singleton"""
        cdn1 = get_cdn_config()
        cdn2 = get_cdn_config()
        assert cdn1 is cdn2
    
    def test_cache_directives_exist(self):
        """Test all cache directives are defined"""
        cdn = CDNConfig()
        assert "static" in cdn.cache_directives
        assert "html" in cdn.cache_directives
        assert "api" in cdn.cache_directives
        assert "dynamic" in cdn.cache_directives
    
    def test_static_assets_directive(self):
        """Test static asset cache directive"""
        cdn = CDNConfig()
        directive = cdn.cache_directives["static"]
        assert "max-age=31536000" in directive  # 1 year
        assert "immutable" in directive
        assert "public" in directive
    
    def test_html_cache_directive(self):
        """Test HTML cache directive"""
        cdn = CDNConfig()
        directive = cdn.cache_directives["html"]
        assert "max-age=3600" in directive  # 1 hour
        assert "must-revalidate" in directive
    
    def test_api_cache_directive(self):
        """Test API cache directive"""
        cdn = CDNConfig()
        directive = cdn.cache_directives["api"]
        assert "max-age=300" in directive  # 5 minutes
    
    def test_dynamic_cache_directive(self):
        """Test dynamic content (no cache) directive"""
        cdn = CDNConfig()
        directive = cdn.cache_directives["dynamic"]
        assert "no-cache" in directive
        assert "no-store" in directive
        assert "max-age=0" in directive


class TestCacheHeaders:
    """Test Cache-Control header generation"""
    
    def test_static_assets_headers(self):
        """Test headers for static assets"""
        cdn = CDNConfig()
        
        paths = [
            "/static/logo.png",
            "/assets/style.css",
            "/js/app.js",
            "/fonts/roboto.woff2",
        ]
        
        for path in paths:
            headers = cdn.get_cache_headers(path)
            assert "Cache-Control" in headers
            assert headers["X-Cache-Policy"] == "STATIC"
            assert "31536000" in headers["Cache-Control"]  # 1 year
    
    def test_html_headers(self):
        """Test headers for HTML pages"""
        cdn = CDNConfig()
        
        paths = ["/", "/index.html", "/dashboard.html"]
        
        for path in paths:
            headers = cdn.get_cache_headers(path)
            assert headers["X-Cache-Policy"] == "HTML"
            assert "3600" in headers["Cache-Control"]  # 1 hour
    
    def test_api_headers(self):
        """Test headers for API responses"""
        cdn = CDNConfig()
        
        paths = ["/api/health", "/v1/metrics", "/api/pipelines"]
        
        for path in paths:
            headers = cdn.get_cache_headers(path)
            assert headers["X-Cache-Policy"] == "API"
            assert "300" in headers["Cache-Control"]  # 5 minutes
    
    def test_dynamic_content_headers(self):
        """Test headers for dynamic content (no cache)"""
        cdn = CDNConfig()
        
        paths = [
            "/pipeline/run",
            "/pipeline/123/result",
            "/streaming/output",
        ]
        
        for path in paths:
            headers = cdn.get_cache_headers(path)
            assert headers["X-Cache-Policy"] == "DYNAMIC"
            assert "no-cache" in headers["Cache-Control"]
            assert "no-store" in headers["Cache-Control"]


class TestCDNConfiguration_Full:
    """Test full CDN configuration"""
    
    def test_get_cdn_configuration(self):
        """Test getting full CDN configuration"""
        cdn = CDNConfig()
        config = cdn.get_cdn_configuration()
        
        assert config["enable_cdn"] is True
        assert config["cache_mode"] == "CACHE_ALL_STATIC"
        assert config["client_ttl"] == 3600
        assert config["max_ttl"] == 31536000
        assert config["negative_caching"] is True
    
    def test_cdn_negative_caching_policy(self):
        """Test negative caching policy for errors"""
        cdn = CDNConfig()
        config = cdn.get_cdn_configuration()
        
        policies = config["negative_caching_policy"]
        assert len(policies) == 3
        
        # 404 should cache for 120 seconds
        assert any(p["code"] == 404 and p["ttl"] == 120 for p in policies)
        # 410 should cache for 120 seconds
        assert any(p["code"] == 410 and p["ttl"] == 120 for p in policies)
        # 501 should cache for 60 seconds
        assert any(p["code"] == 501 and p["ttl"] == 60 for p in policies)


class TestCDNMonitoring:
    """Test CDN monitoring"""
    
    def test_monitoring_initialization(self):
        """Test monitoring initializes correctly"""
        cdn = CDNConfig()
        monitors = cdn.monitoring
        
        assert monitors.metrics["cache_hits"] == 0
        assert monitors.metrics["cache_misses"] == 0
        assert monitors.metrics["total_requests"] == 0
    
    def test_record_cache_hit(self):
        """Test recording cache hit"""
        cdn = CDNConfig()
        monitors = cdn.monitoring
        
        monitors.record_request(from_cache=True, bytes_served=1024)
        
        assert monitors.metrics["cache_hits"] == 1
        assert monitors.metrics["total_requests"] == 1
        assert monitors.metrics["bytes_served_from_cache"] == 1024
    
    def test_record_cache_miss(self):
        """Test recording cache miss"""
        cdn = CDNConfig()
        monitors = cdn.monitoring
        
        monitors.record_request(from_cache=False, bytes_served=2048)
        
        assert monitors.metrics["cache_misses"] == 1
        assert monitors.metrics["total_requests"] == 1
        assert monitors.metrics["bytes_served_from_origin"] == 2048
    
    def test_cache_hit_rate_calculation(self):
        """Test cache hit rate calculation"""
        cdn = CDNConfig()
        monitors = cdn.monitoring
        
        # 4 hits, 1 miss = 80% hit rate
        monitors.record_request(from_cache=True, bytes_served=100)
        monitors.record_request(from_cache=True, bytes_served=100)
        monitors.record_request(from_cache=True, bytes_served=100)
        monitors.record_request(from_cache=True, bytes_served=100)
        monitors.record_request(from_cache=False, bytes_served=100)
        
        hit_rate = monitors.get_cache_hit_rate()
        assert hit_rate == 80.0
    
    def test_bandwidth_saved_calculation(self):
        """Test bandwidth saved calculation"""
        cdn = CDNConfig()
        monitors = cdn.monitoring
        
        # Serve 75% from cache
        monitors.record_request(from_cache=True, bytes_served=750)
        monitors.record_request(from_cache=False, bytes_served=250)
        
        saved = monitors.get_bandwidth_saved()
        assert saved == 75.0
    
    def test_metrics_summary(self):
        """Test metrics summary generation"""
        cdn = CDNConfig()
        monitors = cdn.monitoring
        
        monitors.record_request(from_cache=True, bytes_served=1000)
        monitors.record_request(from_cache=False, bytes_served=500)
        
        summary = monitors.get_metrics_summary()
        
        assert summary["total_requests"] == 2
        assert summary["cache_hits"] == 1
        assert summary["cache_misses"] == 1
        assert summary["cache_hit_rate_percent"] == 50.0
        assert "bandwidth_from_cache_percent" in summary


class TestCDNIntegration:
    """Integration tests for CDN"""
    
    def test_cdn_with_cache_busting(self):
        """Test cache busting with versioned assets"""
        cdn = CDNConfig()
        
        # Versioned assets should still get long cache
        paths = [
            "/static/app.v1.js",
            "/assets/style.v2.css",
        ]
        
        for path in paths:
            headers = cdn.get_cache_headers(path)
            # Contains .js or .css, so treated as static
            assert headers["X-Cache-Policy"] == "STATIC"
    
    def test_cdn_headers_immutability(self):
        """Test that static asset headers include immutable"""
        cdn = CDNConfig()
        
        headers = cdn.get_cache_headers("/static/logo.png")
        assert "immutable" in headers["Cache-Control"]
    
    def test_cdn_error_response_caching(self):
        """Test that error pages are cached briefly"""
        cdn = CDNConfig()
        config = cdn.get_cdn_configuration()
        
        # Verify error caching is enabled
        assert config["negative_caching"] is True
        assert len(config["negative_caching_policy"]) > 0
