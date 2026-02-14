#!/usr/bin/env python
"""
Validation script for UnifiedInferenceGraph implementation.

Verifies:
1. All modules import correctly
2. Core classes are properly defined
3. Preset factory works
4. Graph execution runs without errors
5. Test suite validates
"""

import sys
import traceback
from pathlib import Path

# Add packages to path
packages_dir = Path(__file__).parent / ".." / ".." / "packages" / "aiprod-pipelines" / "src"
sys.path.insert(0, str(packages_dir))


def check_imports():
    """Verify all modules import correctly."""
    print("\n" + "="*60)
    print("✓ CHECKING IMPORTS")
    print("="*60)
    
    try:
        from aiprod_pipelines.inference import (
            GraphNode,
            GraphContext,
            InferenceGraph,
            TextEncodeNode,
            DenoiseNode,
            UpsampleNode,
            DecodeVideoNode,
            AudioEncodeNode,
            CleanupNode,
            PresetFactory,
            preset,
        )
        print("✓ All core classes imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False


def check_graph_context():
    """Verify GraphContext works."""
    print("\n" + "="*60)
    print("✓ TESTING GraphContext")
    print("="*60)
    
    try:
        from aiprod_pipelines.inference import GraphContext
        
        context = GraphContext()
        context["key1"] = "value1"
        context.inputs["key2"] = "value2"
        
        assert context["key1"] == "value1"
        assert context["key2"] == "value2"
        assert "key1" in context
        
        context.update({"key3": "value3"})
        assert context["key3"] == "value3"
        
        print("✓ GraphContext works correctly")
        return True
    except Exception as e:
        print(f"✗ GraphContext test failed: {e}")
        traceback.print_exc()
        return False


def check_graph_node():
    """Verify GraphNode protocol."""
    print("\n" + "="*60)
    print("✓ TESTING GraphNode Protocol")
    print("="*60)
    
    try:
        from aiprod_pipelines.inference import GraphNode, GraphContext
        from typing import Dict, List, Any
        
        # Create test node
        class TestNode(GraphNode):
            @property
            def input_keys(self) -> List[str]:
                return ["test_input"]
            
            @property
            def output_keys(self) -> List[str]:
                return ["test_output"]
            
            def execute(self, context: GraphContext) -> Dict[str, Any]:
                return {"test_output": context["test_input"] * 2}
        
        node = TestNode()
        context = GraphContext()
        context.inputs["test_input"] = 5
        
        result = node.execute(context)
        assert result["test_output"] == 10
        
        print("✓ GraphNode protocol works correctly")
        return True
    except Exception as e:
        print(f"✗ GraphNode test failed: {e}")
        traceback.print_exc()
        return False


def check_inference_graph():
    """Verify InferenceGraph execution."""
    print("\n" + "="*60)
    print("✓ TESTING InferenceGraph DAG Executor")
    print("="*60)
    
    try:
        from aiprod_pipelines.inference import InferenceGraph, GraphNode, GraphContext
        from typing import Dict, List, Any
        
        # Create test nodes
        class InputNode(GraphNode):
            @property
            def input_keys(self) -> List[str]:
                return []
            
            @property
            def output_keys(self) -> List[str]:
                return ["value"]
            
            def execute(self, context: GraphContext) -> Dict[str, Any]:
                return {"value": 5}
        
        class ProcessNode(GraphNode):
            @property
            def input_keys(self) -> List[str]:
                return ["value"]
            
            @property
            def output_keys(self) -> List[str]:
                return ["result"]
            
            def execute(self, context: GraphContext) -> Dict[str, Any]:
                return {"result": context["value"] * 3}
        
        # Build graph
        graph = InferenceGraph("test")
        graph.add_node("input", InputNode())
        graph.add_node("process", ProcessNode())
        graph.connect("input", "process")
        
        # Validate
        is_valid, msg = graph.validate()
        assert is_valid, f"Graph validation failed: {msg}"
        
        # Execute
        result = graph.run()
        assert result["result"] == 15
        
        print("✓ InferenceGraph execution works correctly")
        return True
    except Exception as e:
        print(f"✗ InferenceGraph test failed: {e}")
        traceback.print_exc()
        return False


def check_concrete_nodes():
    """Verify concrete node implementations."""
    print("\n" + "="*60)
    print("✓ TESTING Concrete Nodes")
    print("="*60)
    
    try:
        from aiprod_pipelines.inference import (
            TextEncodeNode,
            DenoiseNode,
            UpsampleNode,
            DecodeVideoNode,
            AudioEncodeNode,
            CleanupNode,
        )
        
        # Check TextEncodeNode
        node = TextEncodeNode(None)
        assert "prompt" in node.input_keys
        assert "embeddings" in node.output_keys
        print("  ✓ TextEncodeNode structure OK")
        
        # Check DenoiseNode
        node = DenoiseNode(None, None)
        assert "latents" in node.input_keys
        assert "embeddings" in node.input_keys
        assert "latents_denoised" in node.output_keys
        print("  ✓ DenoiseNode structure OK")
        
        # Check UpsampleNode
        node = UpsampleNode(None)
        assert "latents" in node.input_keys
        assert "latents_upsampled" in node.output_keys
        print("  ✓ UpsampleNode structure OK")
        
        # Check DecodeVideoNode
        node = DecodeVideoNode(None)
        assert "latents_denoised" in node.input_keys
        assert "video_frames" in node.output_keys
        print("  ✓ DecodeVideoNode structure OK")
        
        # Check AudioEncodeNode
        node = AudioEncodeNode(None)
        assert "audio_prompt" in node.input_keys
        assert "audio_embeddings" in node.output_keys
        print("  ✓ AudioEncodeNode structure OK")
        
        # Check CleanupNode
        node = CleanupNode()
        assert len(node.input_keys) == 0
        assert "memory_freed_mb" in node.output_keys
        print("  ✓ CleanupNode structure OK")
        
        print("✓ All concrete nodes are properly defined")
        return True
    except Exception as e:
        print(f"✗ Concrete nodes test failed: {e}")
        traceback.print_exc()
        return False


def check_presets():
    """Verify preset factory."""
    print("\n" + "="*60)
    print("✓ TESTING Preset Factory")
    print("="*60)
    
    try:
        from aiprod_pipelines.inference import preset, PresetFactory
        
        # Create mock models
        class MockModel:
            def __init__(self, name):
                self.name = name
        
        class MockScheduler:
            timesteps = list(range(30))
        
        mock_encoder = MockModel("encoder")
        mock_model = MockModel("model")
        mock_scheduler = MockScheduler()
        mock_vae = MockModel("vae")
        mock_upsampler = MockModel("upsampler")
        
        # Test each preset mode
        modes = [
            ("t2v_one_stage", False),
            ("t2v_two_stages", True),
            ("distilled_fast", False),
            ("ic_lora", False),
            ("keyframe", False),
        ]
        
        for mode, needs_upsampler in modes:
            try:
                if needs_upsampler:
                    graph = preset(
                        mode,
                        mock_encoder,
                        mock_model,
                        mock_scheduler,
                        mock_vae,
                        upsampler=mock_upsampler,
                    )
                else:
                    graph = preset(
                        mode,
                        mock_encoder,
                        mock_model,
                        mock_scheduler,
                        mock_vae,
                    )
                
                assert graph is not None
                assert graph.name == mode or "t2v" in graph.name or "distilled" in graph.name or "lora" in graph.name
                print(f"  ✓ preset('{mode}') works")
            except Exception as e:
                print(f"  ✗ preset('{mode}') failed: {e}")
                raise
        
        print("✓ All presets factory methods work correctly")
        return True
    except Exception as e:
        print(f"✗ Preset factory test failed: {e}")
        traceback.print_exc()
        return False


def check_test_imports():
    """Verify test modules import correctly."""
    print("\n" + "="*60)
    print("✓ CHECKING TEST IMPORTS")
    print("="*60)
    
    try:
        # Add tests directory
        tests_dir = packages_dir.parent / "tests"
        sys.path.insert(0, str(tests_dir))
        sys.path.insert(0, str(tests_dir / "inference"))
        
        # Try importing test modules
        import conftest
        import test_graph
        import test_nodes
        import test_presets
        import test_integration
        
        print("✓ All test modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Test import failed: {e}")
        traceback.print_exc()
        return False


def print_summary(results):
    """Print validation summary."""
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    
    for check_name, passed_check in results.items():
        status = "✓ PASS" if passed_check else "✗ FAIL"
        print(f"{status}: {check_name}")
    
    print("-" * 60)
    print(f"Result: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 ALL CHECKS PASSED - IMPLEMENTATION IS VALID! 🎉\n")
        return 0
    else:
        print(f"\n❌ {total - passed} CHECK(S) FAILED\n")
        return 1


def main():
    """Run all validation checks."""
    print("\n" + "="*60)
    print("PHASE II INNOVATION 1: UnifiedInferenceGraph")
    print("VALIDATION SUITE")
    print("="*60)
    
    results = {
        "Imports": check_imports(),
        "GraphContext": check_graph_context(),
        "GraphNode Protocol": check_graph_node(),
        "InferenceGraph DAG": check_inference_graph(),
        "Concrete Nodes": check_concrete_nodes(),
        "Preset Factory": check_presets(),
        "Test Imports": check_test_imports(),
    }
    
    exit_code = print_summary(results)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
