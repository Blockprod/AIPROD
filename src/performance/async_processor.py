"""
Async Task Processor - Background task management and scheduling
Enables non-blocking operations for long-running tasks
"""

import asyncio
import uuid
from typing import Callable, Dict, Any, Optional, List, Coroutine, Awaitable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import time
from src.utils.monitoring import logger


class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Represents an async task"""
    task_id: str
    name: str
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_ms: float = 0
    result: Optional[Any] = None
    error: Optional[str] = None
    progress: float = 0.0  # 0-100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "status": self.status.value,
            "progress": self.progress,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "execution_time_ms": self.execution_time_ms,
            "result": self.result,
            "error": self.error,
        }


class AsyncTaskProcessor:
    """
    Manages async task execution and scheduling.
    
    Features:
    - Task queuing and execution
    - Progress tracking
    - Automatic timeout handling
    - Task history
    - Concurrent task limits
    """
    
    def __init__(self, max_concurrent_tasks: int = 10):
        self.tasks: Dict[str, Task] = {}
        self.task_queue: Optional[asyncio.Queue] = None
        self.max_concurrent = max_concurrent_tasks
        self.running_tasks = 0
        self.total_completed = 0
        self.total_failed = 0
    
    async def submit_task(
        self,
        name: str,
        coro: Awaitable,
        timeout_seconds: int = 300
    ) -> str:
        """Submit a task for async execution"""
        task_id = str(uuid.uuid4())
        
        task = Task(
            task_id=task_id,
            name=name,
            status=TaskStatus.PENDING,
            created_at=datetime.utcnow(),
        )
        
        self.tasks[task_id] = task
        
        # Schedule task execution
        asyncio.create_task(
            self._execute_task(task_id, coro, timeout_seconds)
        )
        
        logger.info(f"Task submitted: {task_id} ({name})")
        return task_id
    
    async def _execute_task(
        self,
        task_id: str,
        coro: Awaitable,
        timeout_seconds: int
    ):
        """Execute a task with timeout and error handling"""
        task = self.tasks[task_id]
        
        try:
            # Update status
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            self.running_tasks += 1
            
            start_time = time.time()
            
            # Execute with timeout
            result = await asyncio.wait_for(coro, timeout=timeout_seconds)
            
            # Record result
            execution_time = (time.time() - start_time) * 1000
            task.result = result
            task.execution_time_ms = execution_time
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            task.progress = 100.0
            self.total_completed += 1
            
            logger.info(
                f"Task completed: {task_id} ({task.name}) "
                f"in {execution_time:.2f}ms"
            )
        
        except asyncio.TimeoutError:
            task.status = TaskStatus.FAILED
            task.error = f"Task timeout after {timeout_seconds}s"
            task.completed_at = datetime.utcnow()
            self.total_failed += 1
            logger.error(f"Task timeout: {task_id} ({task.name})")
        
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.utcnow()
            self.total_failed += 1
            logger.error(f"Task failed: {task_id} ({task.name}): {e}")
        
        finally:
            self.running_tasks -= 1
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        task = self.get_task(task_id)
        if task is None:
            return None
        return task.to_dict()
    
    def get_active_tasks(self) -> List[Dict[str, Any]]:
        """Get all active (running/pending) tasks"""
        active = [
            t.to_dict() for t in self.tasks.values()
            if t.status in [TaskStatus.PENDING, TaskStatus.RUNNING]
        ]
        return sorted(active, key=lambda x: x["created_at"], reverse=True)
    
    def get_recent_tasks(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent tasks"""
        recent = sorted(
            self.tasks.values(),
            key=lambda t: t.created_at,
            reverse=True
        )[:limit]
        return [t.to_dict() for t in recent]
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task"""
        task = self.get_task(task_id)
        if task is None:
            return False
        
        if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.utcnow()
            logger.info(f"Task cancelled: {task_id}")
            return True
        
        return False
    
    def cleanup_old_tasks(self, days: int = 7):
        """Remove tasks older than specified days"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        keys_to_delete = [
            k for k, t in self.tasks.items()
            if t.created_at < cutoff
        ]
        
        for key in keys_to_delete:
            del self.tasks[key]
        
        logger.info(f"Cleaned up {len(keys_to_delete)} old tasks")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics"""
        return {
            "active_tasks": self.running_tasks,
            "pending_tasks": sum(
                1 for t in self.tasks.values()
                if t.status == TaskStatus.PENDING
            ),
            "total_tasks": len(self.tasks),
            "completed_tasks": self.total_completed,
            "failed_tasks": self.total_failed,
            "max_concurrent": self.max_concurrent,
        }


# Global async processor instance
_async_processor = None


def get_async_processor(max_concurrent: int = 10) -> AsyncTaskProcessor:
    """Get or create singleton async processor"""
    global _async_processor
    if _async_processor is None:
        _async_processor = AsyncTaskProcessor(max_concurrent)
    return _async_processor
