"""
Mobile Runtime Environment

Runtime environment optimized for mobile and edge inference.
Handles memory management, threading, and platform-specific optimizations.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Callable, Any, List
from enum import Enum
import threading


class MobileRuntime(Enum):
    """Mobile runtime environments."""
    TENSORFLOW_LITE = "tensorflow_lite"
    ONNX_RUNTIME = "onnx_runtime"
    PYTORCH_MOBILE = "pytorch_mobile"
    CORE_ML = "core_ml"  # iOS
    ANDROID_NN = "android_nn"  # Android
    CUSTOM_RUNTIME = "custom_runtime"


@dataclass
class RuntimeMemoryConfig:
    """Memory configuration for runtime."""
    max_memory_mb: int = 512
    enable_memory_pooling: bool = True
    pool_size_mb: int = 256
    enable_gpu_memory: bool = True
    gpu_memory_mb: int = 256
    garbage_collection_interval: int = 100


@dataclass
class RuntimeConfig:
    """Configuration for mobile runtime."""
    runtime_type: MobileRuntime
    memory_config: RuntimeMemoryConfig
    num_threads: int = 4
    enable_gpu: bool = True
    enable_quantization: bool = True
    enable_profiling: bool = False
    
    # Optimization flags
    enable_fused_ops: bool = True
    enable_weight_sharing: bool = True
    enable_activation_caching: bool = True


class InferenceMemoryManager:
    """Manages memory during inference."""
    
    def __init__(self, config: RuntimeMemoryConfig):
        self.config = config
        self.allocated_mb = 0.0
        self.peak_mb = 0.0
        self.lock = threading.Lock()
    
    def allocate(self, size_mb: float) -> bool:
        """Allocate memory."""
        with self.lock:
            if self.allocated_mb + size_mb > self.config.max_memory_mb:
                return False
            
            self.allocated_mb += size_mb
            self.peak_mb = max(self.peak_mb, self.allocated_mb)
            return True
    
    def deallocate(self, size_mb: float) -> None:
        """Deallocate memory."""
        with self.lock:
            self.allocated_mb = max(0.0, self.allocated_mb - size_mb)
    
    def get_utilization_percent(self) -> float:
        """Get memory utilization percentage."""
        with self.lock:
            if self.config.max_memory_mb <= 0:
                return 0.0
            return (self.allocated_mb / self.config.max_memory_mb) * 100


class ThreadPoolExecutor:
    """Thread pool for parallel inference execution."""
    
    def __init__(self, num_threads: int = 4):
        self.num_threads = num_threads
        self.thread_pool: List[threading.Thread] = []
        self.task_queue: List[Callable] = []
        self.lock = threading.Lock()
        self.running = False
    
    def start(self) -> None:
        """Start thread pool."""
        self.running = True
        for _ in range(self.num_threads):
            t = threading.Thread(target=self._worker)
            t.daemon = True
            t.start()
            self.thread_pool.append(t)
    
    def _worker(self) -> None:
        """Worker thread function."""
        while self.running:
            task = None
            with self.lock:
                if self.task_queue:
                    task = self.task_queue.pop(0)
            
            if task:
                task()
    
    def submit_task(self, task: Callable) -> None:
        """Submit task to pool."""
        with self.lock:
            self.task_queue.append(task)
    
    def shutdown(self) -> None:
        """Shutdown thread pool."""
        self.running = False
        for t in self.thread_pool:
            t.join(timeout=1.0)


class MobileInferenceRuntime:
    """Runtime for mobile inference execution."""
    
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.memory_manager = InferenceMemoryManager(config.memory_config)
        self.thread_pool = ThreadPoolExecutor(config.num_threads)
        self.inference_count = 0
        self.total_latency_ms = 0.0
    
    def initialize(self) -> bool:
        """Initialize runtime."""
        self.thread_pool.start()
        return True
    
    def execute_inference(
        self,
        model_input: Dict[str, Any],
        inference_fn: Callable,
    ) -> Dict[str, Any]:
        """Execute inference on runtime."""
        # Allocate memory
        estimated_memory = 100.0  # MB
        if not self.memory_manager.allocate(estimated_memory):
            return {"error": "Out of memory"}
        
        try:
            # Run inference
            import time
            start_time = time.time()
            result = inference_fn(model_input)
            latency_ms = (time.time() - start_time) * 1000
            
            self.inference_count += 1
            self.total_latency_ms += latency_ms
            
            return {
                "result": result,
                "latency_ms": latency_ms,
                "memory_util_percent": self.memory_manager.get_utilization_percent(),
            }
        finally:
            self.memory_manager.deallocate(estimated_memory)
    
    def get_average_latency_ms(self) -> float:
        """Get average inference latency."""
        if self.inference_count == 0:
            return 0.0
        return self.total_latency_ms / self.inference_count
    
    def shutdown(self) -> None:
        """Shutdown runtime."""
        self.thread_pool.shutdown()
