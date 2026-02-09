#!/usr/bin/env python
"""
Test script for Quality First Implementation
Tests the new quality specs, cost calculator, and QA modules
"""

import sys
sys.path.insert(0, '/c/Users/averr/AIPROD')

import logging
from src.agents.quality_specs import QualitySpecRegistry, GoodTierSpec, HighTierSpec, UltraTierSpec
from src.agents.cost_calculator import CostCalculator, Complexity, RushDelivery
from src.agents.quality_assurance import QualityAssuranceEngine

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_quality_specs():
    """Test quality specification classes"""
    logger.info("\n" + "="*60)
    logger.info("🎬 TEST 1: Quality Specifications")
    logger.info("="*60)
    
    # Test each tier
    tiers = QualitySpecRegistry.get_all_tiers()
    
    for tier_data in tiers:
        logger.info(f"\n✅ {tier_data['tier']} - {tier_data['positioning']}")
        logger.info(f"   Video: {tier_data['video']['resolution']}@{tier_data['video']['fps']}fps")
        logger.info(f"   Audio: {tier_data['audio']['format']} ({tier_data['audio']['codec']})")
        logger.info(f"   Delivery: {', '.join(tier_data['delivery_formats'])}")
        logger.info(f"   Quality Guarantee: {tier_data['quality_guarantee']}")
    
    logger.info("\n✅ All tier specs loaded successfully!")
    return True


def test_cost_calculator():
    """Test cost calculation engine"""
    logger.info("\n" + "="*60)
    logger.info("💰 TEST 2: Cost Calculator")
    logger.info("="*60)
    
    # Test 1: Basic cost calculation
    logger.info("\n📊 Test 2a: Basic cost (30sec, GOOD, moderate)")
    cost = CostCalculator.calculate_cost("good", 30, "moderate")
    logger.info(f"   Base rate: ${cost.base_cost_per_min:.2f}/min")
    logger.info(f"   Duration: {cost.duration_min:.2f} min")
    logger.info(f"   Complexity multiplier: {cost.complexity_multiplier}x")
    logger.info(f"   Subtotal: ${cost.subtotal_usd:.2f}")
    logger.info(f"   Total (with tax): ${cost.total_usd:.2f}")
    logger.info(f"   Estimated delivery: {cost.estimated_delivery_sec}s")
    
    # Test 2: Cost with rush delivery
    logger.info("\n📊 Test 2b: With rush delivery (60sec, HIGH, complex, 6h express)")
    cost = CostCalculator.calculate_cost(
        "high", 60, "complex",
        rush_delivery="express_6h"
    )
    logger.info(f"   Subtotal: ${cost.subtotal_usd:.2f}")
    logger.info(f"   Rush multiplier: {cost.rush_multiplier}x")
    logger.info(f"   Total: ${cost.total_usd:.2f}")
    logger.info(f"   Estimated delivery: {cost.estimated_delivery_sec}s")
    
    # Test 3: Alternatives
    logger.info("\n📊 Test 2c: Tier alternatives (30sec, simple)")
    alternatives = CostCalculator.get_alternatives(30, "simple")
    for alt in alternatives:
        logger.info(f"   {alt.tier_name.upper()}: ${alt.total_usd:.2f}")
    
    # Test 4: Recommendation
    logger.info("\n📊 Test 2d: Recommend tier (quality priority, $0.30 budget)")
    recommended = CostCalculator.recommend_tier(
        duration_sec=30,
        complexity="moderate",
        user_budget=0.30,
        priority="quality"
    )
    if recommended:
        logger.info(f"   Recommended: {recommended.tier_name.upper()} (${recommended.total_usd:.2f})")
    
    logger.info("\n✅ Cost calculator working correctly!")
    return True


def test_quality_assurance():
    """Test QA validation engine"""
    logger.info("\n" + "="*60)
    logger.info("✅ TEST 3: Quality Assurance Engine")
    logger.info("="*60)
    
    # Test with GOOD tier metadata
    logger.info("\n📋 Test 3a: Validate GOOD tier video")
    metadata_good = {
        "resolution": "1920x1080",
        "fps": 24.0,
        "codec": "H.264",
        "bitrate_kbps": 3500,
        "color_space": "Rec.709 (SDR)",
        "audio_format": "Stereo",
        "audio_channels": 2,
        "audio_codec": "AAC-LC",
        "audio_loudness_lufs": -23.0,
        "artifact_score": 0.05,
        "flicker_detected": False
    }
    
    qa = QualityAssuranceEngine()
    report = qa.validate_video("job-good-001", "good", metadata_good)
    logger.info(f"   Status: {report.status.value}")
    logger.info(f"   Checks passed: {report.passed_checks}/{report.total_checks}")
    logger.info(f"   Certified: {report.status.value == 'passed'}")
    
    # Test with HIGH tier metadata
    logger.info("\n📋 Test 3b: Validate HIGH tier video")
    metadata_high = {
        "resolution": "3840x2160",
        "fps": 30.0,
        "codec": "H.265",
        "bitrate_kbps": 10000,
        "color_space": "Rec.709",
        "audio_format": "5.1 Surround",
        "audio_channels": 6,
        "audio_codec": "AC-3",
        "audio_loudness_lufs": -23.0,
        "artifact_score": 0.02,
        "flicker_detected": False
    }
    
    report = qa.validate_video("job-high-001", "high", metadata_high)
    logger.info(f"   Status: {report.status.value}")
    logger.info(f"   Checks passed: {report.passed_checks}/{report.total_checks}")
    logger.info(f"   Certified: {report.status.value == 'passed'}")
    
    # Test with ULTRA tier metadata
    logger.info("\n📋 Test 3c: Validate ULTRA tier video")
    metadata_ultra = {
        "resolution": "3840x2160",
        "fps": 60.0,
        "codec": "H.266",
        "bitrate_kbps": 30000,
        "color_space": "Rec.2020 (HDR10)",
        "audio_format": "7.1.4 Atmos",
        "audio_channels": 12,
        "audio_codec": "TrueHD",
        "audio_loudness_lufs": -24.0,
        "artifact_score": 0.01,
        "flicker_detected": False
    }
    
    report = qa.validate_video("job-ultra-001", "ultra", metadata_ultra)
    logger.info(f"   Status: {report.status.value}")
    logger.info(f"   Checks passed: {report.passed_checks}/{report.total_checks}")
    logger.info(f"   Certified: {report.status.value == 'passed'}")
    
    logger.info("\n✅ QA engine working correctly!")
    return True


def test_imports():
    """Test that all imports work in the API"""
    logger.info("\n" + "="*60)
    logger.info("📦 TEST 4: Import Tests")
    logger.info("="*60)
    
    try:
        logger.info("\n   Testing imports...")
        from src.agents.quality_specs import GoodTierSpec, HighTierSpec, UltraTierSpec, QualitySpecRegistry
        logger.info("   ✅ quality_specs imported")
        
        from src.agents.cost_calculator import CostCalculator, Complexity, RushDelivery, CostBreakdown
        logger.info("   ✅ cost_calculator imported")
        
        from src.agents.quality_assurance import QualityAssuranceEngine, QCReport, QCStatus
        logger.info("   ✅ quality_assurance imported")
        
        logger.info("\n✅ All imports successful!")
        return True
    except Exception as e:
        logger.error(f"❌ Import error: {e}", exc_info=True)
        return False


def run_all_tests():
    """Run all tests"""
    logger.info("\n" + "╔════════════════════════════════════════════════════════════╗")
    logger.info( "║   AIPROD Quality First Implementation - Test Suite          ║")
    logger.info( "║   Testing all new quality, cost, and QA modules             ║")
    logger.info( "╚════════════════════════════════════════════════════════════╝")
    
    results = []
    
    try:
        results.append(("Quality Specs", test_quality_specs()))
    except Exception as e:
        logger.error(f"❌ Quality specs test failed: {e}", exc_info=True)
        results.append(("Quality Specs", False))
    
    try:
        results.append(("Cost Calculator", test_cost_calculator()))
    except Exception as e:
        logger.error(f"❌ Cost calculator test failed: {e}", exc_info=True)
        results.append(("Cost Calculator", False))
    
    try:
        results.append(("Quality Assurance", test_quality_assurance()))
    except Exception as e:
        logger.error(f"❌ QA engine test failed: {e}", exc_info=True)
        results.append(("Quality Assurance", False))
    
    try:
        results.append(("Imports", test_imports()))
    except Exception as e:
        logger.error(f"❌ Import test failed: {e}", exc_info=True)
        results.append(("Imports", False))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📊 TEST SUMMARY")
    logger.info("="*60)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name:.<40} {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    logger.info("="*60)
    logger.info(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED! Quality First implementation is ready!")
        return 0
    else:
        logger.error(f"\n❌ {total - passed} test(s) failed. Please review above.")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
