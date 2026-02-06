#!/usr/bin/env python3
"""
Debug RenderExecutor API key loading
"""

import os
import sys

print('🔍 DEBUGGING RENDEREXECUTOR API KEY LOADING')
print('=' * 70)

# Check environment
runway_key = os.getenv('RUNWAY_API_KEY', '<NOT SET>')
print(f'1. os.getenv("RUNWAY_API_KEY"): {runway_key[:50] if runway_key != "<NOT SET>" else runway_key}...')
print(f'   Length: {len(runway_key) if runway_key != "<NOT SET>" else 0}')

# Now import and create RenderExecutor
try:
    from src.agents.render_executor import RenderExecutor
    executor = RenderExecutor()
    print(f'\n2. RenderExecutor.runway_api_key: {executor.runway_api_key[:50] if executor.runway_api_key else "EMPTY"}...')
    print(f'   Length: {len(executor.runway_api_key) if executor.runway_api_key else 0}')
    
    # Check the condition that forces mock
    forces_mock = not executor.runway_api_key or executor.runway_api_key == 'your-runway-api-key'
    
    print(f'\n3. Mock Condition Check:')
    print(f'   not runway_api_key: {not executor.runway_api_key}')
    print(f'   == placeholder: {executor.runway_api_key == "your-runway-api-key"}')
    print(f'   FORCES MOCK: {forces_mock}')
    
    if forces_mock:
        print(f'\n❌ RESULT: Will return MOCK videos')
        print(f'   Why: API key is empty or placeholder')
    else:
        print(f'\n✅ RESULT: Will use REAL Runway ML')
        print(f'   API key is loaded and valid!')
        
except Exception as e:
    print(f'❌ Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
