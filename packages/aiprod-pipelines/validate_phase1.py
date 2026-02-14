#!/usr/bin/env python
"""
Quick validation that PHASE 1 adapters load correctly.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "aiprod-pipelines" / "src"))

# Test loading adapters
outcomes = []

# Test 1: Base adapter
try:
    with open(project_root / "aiprod-pipelines" / "src" / "aiprod_pipelines" / "api" / "adapters" / "base.py") as f:
        code = f.read()
    namespace = {}
    exec(code, namespace)
    outcomes.append(("BaseAdapter", True, None))
except Exception as e:
    outcomes.append(("BaseAdapter", False, str(e)))

# Test 2: Input Sanitizer
try:
    base_code = open(project_root / "aiprod-pipelines" / "src" / "aiprod_pipelines" / "api" / "adapters" / "base.py").read()
    base_ns = {}
    exec(base_code, base_ns)
    
    sanitizer_code = open(project_root / "aiprod-pipelines" / "src" / "aiprod_pipelines" / "api" / "adapters" / "input_sanitizer.py").read()
    sanitizer_code = sanitizer_code.replace("from ..schema.schemas import Context", "Context = dict")
    sanitizer_code = sanitizer_code.replace("from .base import BaseAdapter", "")
    
    sanitizer_ns = {"BaseAdapter": base_ns["BaseAdapter"], "Dict": dict, "Any": object, "Optional": type(None)}
    exec(sanitizer_code, sanitizer_ns)
    outcomes.append(("InputSanitizerAdapter", True, None))
except Exception as e:
    outcomes.append(("InputSanitizerAdapter", False, str(e)))

# Test 3: Creative Director
try:
    creative_code = open(project_root / "aiprod-pipelines" / "src" / "aiprod_pipelines" / "api" / "adapters" / "creative.py").read()
    creative_code = creative_code.replace("from ..schema.schemas import Context", "Context = dict")
    creative_code = creative_code.replace("from .base import BaseAdapter", "")
    creative_code = creative_code.replace("import google.generativeai as genai", "genai = None")
    
    import asyncio
    import time
    import random
    creative_ns = {
        "BaseAdapter": base_ns["BaseAdapter"],
        "Context": dict,
        "Dict": dict, "Any": object, "Optional": type(None), "List": list,
        "asyncio": asyncio, "time": time, "random": random,
        "Tuple": tuple, "Callable": callable
    }
    exec(creative_code, creative_ns)
    outcomes.append(("CreativeDirectorAdapter", True, None))
except Exception as e:
    outcomes.append(("CreativeDirectorAdapter", False, str(e)))

# Test 4: Visual Translator
try:
    visual_code = open(project_root / "aiprod-pipelines" / "src" / "aiprod_pipelines" / "api" / "adapters" / "visual_translator.py").read()
    visual_code = visual_code.replace("from ..schema.schemas import Context", "Context = dict")
    visual_code = visual_code.replace("from .base import BaseAdapter", "")
    
    import hashlib
    visual_ns = {
        "BaseAdapter": base_ns["BaseAdapter"],
        "Context": dict,
        "Dict": dict, "Any": object, "Optional": type(None), "List": list, "Tuple": tuple,
        "hashlib": hashlib
    }
    exec(visual_code, visual_ns)
    outcomes.append(("VisualTranslatorAdapter", True, None))
except Exception as e:
    outcomes.append(("VisualTranslatorAdapter", False, str(e)))

# Test 5: Render Executor
try:
    render_code = open(project_root / "aiprod-pipelines" / "src" / "aiprod_pipelines" / "api" / "adapters" / "render.py").read()
    render_code = render_code.replace("from ..schema.schemas import Context", "Context = dict")
    render_code = render_code.replace("from .base import BaseAdapter", "")
    
    render_ns = {
        "BaseAdapter": base_ns["BaseAdapter"],
        "Context": dict,
        "Dict": dict, "Any": object, "Optional": type(None), "List": list, "Tuple": tuple,
        "asyncio": asyncio, "time": time, "random": random
    }
    exec(render_code, render_ns)
    outcomes.append(("RenderExecutorAdapter", True, None))
except Exception as e:
    outcomes.append(("RenderExecutorAdapter", False, str(e)))

# Print results
print("\n" + "=" * 70)
print("PHASE 1 ADAPTER LOAD TEST")
print("=" * 70)

for name, success, error in outcomes:
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"\n{status}: {name}")
    if error:
        print(f"  Error: {error[:100]}")

passed = sum(1 for _, s, _ in outcomes if s)
total = len(outcomes)
print(f"\n{passed}/{total} adapters loaded successfully")
print("=" * 70)
