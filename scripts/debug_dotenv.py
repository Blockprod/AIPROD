#!/usr/bin/env python3
"""
Debug why load_dotenv() might not be working in RenderExecutor
"""

import os
import sys
from pathlib import Path

print("Debugging load_dotenv() path resolution:")
print(f"Current working directory: {os.getcwd()}")
print()

# Check if .env exists in current directory
env_path_current = Path('.env')
print(f".env in current directory: {env_path_current.exists()}")

# Check if .env exists in workspace root
workspace_root = Path('c:\\Users\\averr\\AIPROD')
env_path_abs = workspace_root / '.env'
print(f".env in workspace root: {env_path_abs.exists()} - {env_path_abs}")

# Now try load_dotenv
print("\nTrying load_dotenv()...")
from dotenv import load_dotenv

# Try with no arguments (should use .env in current directory)
result1 = load_dotenv(override=False)
print(f"load_dotenv() with no args: {result1}")

# Try with absolute path
result2 = load_dotenv(env_path_abs, override=False)
print(f"load_dotenv(absolute_path): {result2}")

# Check if keys are loaded
runway = os.getenv('RUNWAY_API_KEY')  
gemini = os.getenv('GEMINI_API_KEY')

print(f"\nAfter load_dotenv:")
print(f"RUNWAY_API_KEY: {runway[:30] + '...' if runway else 'NOT SET'}")
print(f"GEMINI_API_KEY: {gemini[:30] + '...' if gemini else 'NOT SET'}")

# Now test RenderExecutor
print("\n" + "="*70)
print("Testing RenderExecutor with loaded env:")
print("="*70)
try:
    from src.agents.render_executor import RenderExecutor
    executor = RenderExecutor()
    print(f"RenderExecutor.runway_api_key: {executor.runway_api_key[:30] + '...' if executor.runway_api_key else 'EMPTY'}")
except Exception as e:
    print(f"Error creating RenderExecutor: {e}")
    import traceback
    traceback.print_exc()
