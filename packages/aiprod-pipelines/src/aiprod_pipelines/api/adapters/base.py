"""
Base Adapter Protocols and Abstract Classes
===========================================

Defines the contract that all adapters must implement.
"""

from typing import Protocol, Dict, Any
from abc import ABC, abstractmethod
from ..schema.schemas import Context


class AdapterProtocol(Protocol):
    """Protocol that all adapters must implement."""
    
    async def execute(self, ctx: Context) -> Context:
        """
        Execute adapter logic and return updated context.
        
        Args:
            ctx: Current execution context
            
        Returns:
            Updated execution context
        """
        ...


class BaseAdapter(ABC):
    """
    Base class for all adapters.
    
    Provides common functionality and enforces the adapter contract.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize adapter.
        
        Args:
            config: Adapter-specific configuration
        """
        self.config = config or {}
        self.name = self.__class__.__name__
    
    @abstractmethod
    async def execute(self, ctx: Context) -> Context:
        """
        Execute adapter logic.
        
        Must be implemented by all concrete adapters.
        
        Args:
            ctx: Current execution context
            
        Returns:
            Updated execution context
        """
        raise NotImplementedError(f"{self.name}.execute() must be implemented")
    
    def validate_context(self, ctx: Context, required_keys: list) -> bool:
        """
        Validate that context contains required keys.
        
        Args:
            ctx: Context to validate
            required_keys: List of required keys in ctx["memory"]
            
        Returns:
            True if valid, False otherwise
        """
        memory = ctx.get("memory", {})
        
        for key in required_keys:
            if key not in memory:
                return False
        
        return True
    
    def log(self, level: str, message: str, **kwargs):
        """
        Log adapter activity.
        
        Args:
            level: Log level (info, warning, error)
            message: Log message
            **kwargs: Additional context
        """
        # In production, would use proper logging
        print(f"[{self.name}] {level.upper()}: {message}", kwargs)
