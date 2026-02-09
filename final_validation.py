#!/usr/bin/env python3
"""Final validation test for P0.2 & P0.3 endpoints"""

import json
import sys
from src.api.cost_estimator import CostEstimator

print("\n" + "="*70)
print("FINAL VALIDATION: P0.2 & P0.3 Video Endpoints")
print("="*70)

# Test 1: CostEstimator
print("\n1. Testing CostEstimator class...")
try:
    estimator = CostEstimator()
    plans = estimator.estimate_plans("Test prompt", 100.0, {})
    
    print(f"   Generated {len(plans)} plans:")
    total_valid = 0
    for plan in plans:
        plan_dict = plan.to_dict()
        print(f"   - {plan_dict['tier']:8} | ${plan_dict['estimated_cost']:6.2f} | {plan_dict['resolution']:15} | {plan_dict['backend']}")
        total_valid += 1
        
        # Validate required fields
        required_fields = ['tier', 'estimated_cost', 'time_seconds', 'quality', 'resolution', 'backend']
        for field in required_fields:
            if field not in plan_dict:
                print(f"      MISSING FIELD: {field}")
                total_valid -= 1
    
    if total_valid == 3:
        print("   STATUS: PASS")
    else:
        print("   STATUS: FAIL")
        sys.exit(1)
        
except Exception as e:
    print(f"   STATUS: FAIL - {str(e)[:100]}")
    sys.exit(1)

# Test 2: Verify endpoints exist
print("\n2. Checking endpoints in FastAPI app...")
try:
    from src.api.main import app
    
    routes = [route.path for route in app.routes if hasattr(route, 'path')]
    
    video_routes = [r for r in routes if 'video' in r]
    print(f"   Found {len(video_routes)} video routes:")
    for route in video_routes:
        print(f"   - {route}")
    
    if '/video/plan' in routes and '/video/generate' in routes:
        print("   STATUS: PASS")
    else:
        print("   STATUS: FAIL - Missing endpoints")
        sys.exit(1)
        
except Exception as e:
    print(f"   STATUS: FAIL - {str(e)[:100]}")
    sys.exit(1)

# Summary
print("\n" + "="*70)
print("VALIDATION COMPLETE: Ready for P0.4")
print("="*70)
print("\nNext Steps:")
print("  1. Create React dashboard (P0.4)")
print("  2. Wire frontend to /video/plan and /video/generate")
print("  3. Deploy to production")
print("\n" + "="*70 + "\n")
