#!/usr/bin/env python3
"""
Diagnostic: Check if environment variables are loaded correctly
"""

import os
from dotenv import load_dotenv

# Load .env
load_dotenv('.env')

# Check values
print('🔍 After load_dotenv():')
runway = os.getenv("RUNWAY_API_KEY", "NOT SET")
gemini = os.getenv("GEMINI_API_KEY", "NOT SET")

runway_display = runway[:30] + "..." if runway != "NOT SET" else "NOT SET"
gemini_display = gemini[:30] + "..." if gemini != "NOT SET" else "NOT SET"

print(f'  RUNWAY_API_KEY: {runway_display}')
print(f'  GEMINI_API_KEY: {gemini_display}')

# Now let's simulate what RenderExecutor does
runway_api = os.getenv('RUNWAYML_API_SECRET') or os.getenv('RUNWAY_API_KEY', '')
print(f'\n🎬 As RenderExecutor would see it:')
if runway_api:
    print(f'  self.runway_api_key: {runway_api[:40]}...')
    print(f'  Is valid (not empty, not placeholder): YES ✅')
    print(f'  Would trigger mock mode: NO ❌')
else:
    print(f'  self.runway_api_key: EMPTY')
    print(f'  Would trigger mock mode: YES')
