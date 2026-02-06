#!/usr/bin/env python3
"""
Test RenderExecutor directly with simulated API call
"""

import asyncio
import sys
import io

# Fix encoding on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer,  encoding='utf-8')

from dotenv import load_dotenv
load_dotenv(override=True)

print("Starting RenderExecutor test...")
print("="*70)

from src.agents.render_executor import RenderExecutor

# Create executor
executor = RenderExecutor()
print(f"CreatedRenderExecutor")
print(f"  runway_api_key: {executor.runway_api_key[:40] + '...' if executor.runway_api_key else 'EMPTY'}")

# Create a simple prompt  
prompt_bundle = {
    "text_prompt": "A beautiful sunset",
    "quality_required": 0.8
}

# Run async
print(f"\nCalling executor.run()...")
result = asyncio.run(executor.run(prompt_bundle))

print(f"\nResult:")
print(f"  backend: {result.get('backend')}")
print(f"  status: {result.get('status')}")
print(f"  video_url: {result.get('video_url')}")

if result.get('backend') == 'mock':
    print(f"\n⚠️  Still mock - runway_api_key in executor.run():")
    # Call run again to trigger the reload logging
    message = "RenderExecutor will reload .env in run() method and log the key status"
    print(f"  Message: {message}")
else:
    print(f"\n✨ REAL BACKEND USED!")
