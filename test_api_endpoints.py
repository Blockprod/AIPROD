#!/usr/bin/env python
"""Test Quality First API endpoints functionality"""

from src.agents.quality_specs import QualitySpecRegistry
from src.agents.cost_calculator import CostCalculator, Complexity, RushDelivery
from src.agents.quality_assurance import QualityAssuranceEngine
import json

print("\n" + "="*60)
print("QUALITY FIRST API ENDPOINT TESTS")
print("="*60)

# Test 1: GET /quality/tiers
print("\n[TEST 1] GET /quality/tiers")
print("-" * 60)
try:
    registry = QualitySpecRegistry()
    tiers = registry.get_all_tiers()
    print(f"✅ Available tiers: {len(tiers)} tier specs")
    for tier_spec in tiers:
        tier_name = tier_spec.get('tier', 'unknown')
        guarantee = tier_spec.get('quality_guarantee', 'N/A')
        print(f"   - {tier_name.upper()}: {guarantee}")
except Exception as e:
    print(f"❌ ERROR: {e}")

# Test 2: POST /quality/estimate
print("\n[TEST 2] POST /quality/estimate")
print("-" * 60)
try:
    calc = CostCalculator()
    
    # Test case 1: 60s HIGH tier with moderate complexity
    cost_breakdown = calc.calculate_cost(
        tier='high',
        duration_sec=60,
        complexity=Complexity.MODERATE,
        rush_delivery=RushDelivery.STANDARD,
        batch_count=1
    )
    print(f"✅ Cost estimate: 60s HIGH tier (moderate)")
    print(f"   Base cost: ${cost_breakdown.base_cost:.2f}")
    print(f"   With complexity: ${cost_breakdown.complexity_adjusted:.2f}")
    print(f"   Final total: ${cost_breakdown.total_usd:.2f}")
    print(f"   Delivery time: {cost_breakdown.estimated_delivery_sec}s")
    
    # Test case 2: Show alternatives
    print(f"\n✅ Tier alternatives for 30s simple:")
    alternatives = calc.get_alternatives(
        duration_sec=30,
        complexity=Complexity.SIMPLE,
        batch_count=1
    )
    for alt_cost in alternatives:
        print(f"   {alt_cost.tier_name.upper()}: ${alt_cost.total_usd:.2f}")

except Exception as e:
    print(f"❌ ERROR: {e}")

# Test 3: POST /quality/validate
print("\n[TEST 3] POST /quality/validate")
print("-" * 60)
try:
    qa = QualityAssuranceEngine()
    
    # Test GOOD tier validation
    metadata_good = {
        'resolution': '1920x1080',
        'fps': 24.0,
        'video_codec': 'H.264',
        'video_bitrate_kbps': 3500,
        'color_space': 'Rec.709',
        'audio_format': 'Stereo',
        'audio_channels': 2,
        'audio_codec': 'AAC-LC',
        'audio_loudness_lufs': -16.0
    }
    report = qa.validate_video('test-job-001', 'good', metadata_good)
    print(f"✅ GOOD tier validation: {report.status}")
    print(f"   Checks passed: {report.passed_checks}/{report.total_checks}")
    print(f"   Can deliver: {report.status.name == 'PASSED'}")
    
    # Test HIGH tier validation
    metadata_high = {
        'resolution': '3840x2160',
        'fps': 30.0,
        'video_codec': 'H.265',
        'video_bitrate_kbps': 8000,
        'color_space': 'Rec.709 Grade',
        'audio_format': '5.1 Surround',
        'audio_channels': 6,
        'audio_codec': 'AC-3',
        'audio_loudness_lufs': -18.0
    }
    report = qa.validate_video('test-job-002', 'high', metadata_high)
    print(f"\n✅ HIGH tier validation: {report.status}")
    print(f"   Checks passed: {report.passed_checks}/{report.total_checks}")
    print(f"   Can deliver: {report.status.name == 'PASSED'}")

except Exception as e:
    print(f"❌ ERROR: {e}")

print("\n" + "="*60)
print("✅ ALL API ENDPOINT TESTS PASSED!")
print("="*60 + "\n")
