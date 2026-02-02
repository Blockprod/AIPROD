import pytest
import time
import asyncio
from src.orchestrator.state_machine import StateMachine

@pytest.mark.asyncio
async def test_pipeline_performance_latency():
    """
    Test de performance : latence du pipeline complet.
    """
    sm = StateMachine()
    inputs = {"priority": "high", "lang": "en"}
    start = time.time()
    result = await sm.run(inputs)
    latency_ms = (time.time() - start) * 1000
    # Target : < 20 secondes (20000ms) pour Fast Track
    assert latency_ms < 20000, f"Latency too high: {latency_ms}ms"


@pytest.mark.asyncio
async def test_pipeline_performance_multiple_runs():
    """
    Test de performance : nombre de runs parallèles.
    """
    sm = StateMachine()
    tasks = []
    for i in range(3):
        inputs = {"priority": "high", "lang": "en"}
        tasks.append(sm.run(inputs))
    
    start = time.time()
    results = await asyncio.gather(*tasks)
    latency_ms = (time.time() - start) * 1000
    
    assert len(results) == 3
    assert all(r is not None for r in results)
    # Latency should be reasonable for parallel execution
    assert latency_ms < 60000
