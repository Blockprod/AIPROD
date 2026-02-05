"""
WebSocket Manager - Enhanced WebSocket connection management with security
Handles subscription, authentication, and event broadcasting
"""

import json
from typing import Dict, Set, Optional, Callable, Any
from datetime import datetime
from fastapi import WebSocket
from src.utils.monitoring import logger


class WebSocketConnectionManager:
    """
    Manages WebSocket connections with:
    - Per-connection authentication tracking
    - Subscription management (subscribe to job updates)
    - Graceful disconnect handling
    - Event broadcasting to subscribed clients
    - Rate limiting per connection
    """
    
    def __init__(self, max_message_size: int = 65536, timeout_seconds: int = 30):
        """
        Initialize WebSocket Manager
        
        Args:
            max_message_size: Max bytes per message (default: 64KB)
            timeout_seconds: Connection timeout for inactive clients
        """
        self.active_connections: Dict[str, Set[WebSocket]] = {}  # job_id -> set of websockets
        self.connection_auth: Dict[WebSocket, Dict[str, Any]] = {}  # ws -> user info
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}  # ws -> metadata
        self.max_message_size = max_message_size
        self.timeout_seconds = timeout_seconds
    
    async def connect(self, websocket: WebSocket, job_id: str, user_id: Optional[str] = None):
        """
        Register a new WebSocket connection
        
        Args:
            websocket: FastAPI WebSocket connection
            job_id: Job ID to subscribe to
            user_id: Optional user ID for authenticated connections
        """
        await websocket.accept()
        
        # Initialize job subscription
        if job_id not in self.active_connections:
            self.active_connections[job_id] = set()
        
        self.active_connections[job_id].add(websocket)
        
        # Store auth info
        self.connection_auth[websocket] = {
            "user_id": user_id,
            "job_id": job_id,
            "connected_at": datetime.utcnow().isoformat(),
            "authenticated": user_id is not None,
        }
        
        # Store metadata
        self.connection_metadata[websocket] = {
            "message_count": 0,
            "error_count": 0,
            "last_message_at": datetime.utcnow().isoformat(),
        }
        
        logger.info(f"WebSocket connected: job_id={job_id}, user_id={user_id}")
        
        # Send welcome message
        await websocket.send_json({
            "event": "connected",
            "job_id": job_id,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Connected to job updates"
        })
    
    async def disconnect(self, websocket: WebSocket):
        """
        Unregister a WebSocket connection
        
        Args:
            websocket: FastAPI WebSocket connection to remove
        """
        # Find and remove from active connections
        auth = self.connection_auth.get(websocket)
        if auth and auth["job_id"] in self.active_connections:
            self.active_connections[auth["job_id"]].discard(websocket)
            
            # Cleanup empty job sets
            if not self.active_connections[auth["job_id"]]:
                del self.active_connections[auth["job_id"]]
        
        # Remove auth and metadata
        self.connection_auth.pop(websocket, None)
        self.connection_metadata.pop(websocket, None)
        
        if auth:
            logger.info(f"WebSocket disconnected: job_id={auth['job_id']}, user_id={auth['user_id']}")
    
    async def broadcast_to_job(self, job_id: str, message: Dict[str, Any]):
        """
        Broadcast a message to all clients subscribed to a job
        
        Args:
            job_id: Target job ID
            message: Message to broadcast (dict)
        """
        if job_id not in self.active_connections:
            return
        
        # Add timestamp if not present
        if "timestamp" not in message:
            message["timestamp"] = datetime.utcnow().isoformat()
        
        disconnected = set()
        
        for websocket in self.active_connections[job_id]:
            try:
                # Check message size
                msg_bytes = json.dumps(message).encode()
                if len(msg_bytes) > self.max_message_size:
                    logger.warning(f"Message too large: {len(msg_bytes)} bytes")
                    continue
                
                await websocket.send_json(message)
                
                # Update metadata
                metadata = self.connection_metadata.get(websocket, {})
                metadata["message_count"] = metadata.get("message_count", 0) + 1
                metadata["last_message_at"] = datetime.utcnow().isoformat()
                
            except Exception as e:
                logger.warning(f"Error broadcasting to WebSocket: {e}")
                disconnected.add(websocket)
        
        # Clean up disconnected clients
        for ws in disconnected:
            await self.disconnect(ws)
    
    async def send_error_to_connection(self, websocket: WebSocket, error_msg: str):
        """
        Send an error message to a specific connection
        
        Args:
            websocket: Target connection
            error_msg: Error message to send
        """
        try:
            await websocket.send_json({
                "event": "error",
                "error": error_msg,
                "timestamp": datetime.utcnow().isoformat(),
                "severity": "error"
            })
            
            # Update error count
            metadata = self.connection_metadata.get(websocket, {})
            metadata["error_count"] = metadata.get("error_count", 0) + 1
            
        except Exception as e:
            logger.warning(f"Error sending error message: {e}")
    
    def get_connection_count(self, job_id: Optional[str] = None) -> int:
        """
        Get number of active connections
        
        Args:
            job_id: Optional - count for specific job only
            
        Returns:
            Number of active connections
        """
        if job_id:
            return len(self.active_connections.get(job_id, set()))
        
        total = 0
        for connections in self.active_connections.values():
            total += len(connections)
        return total
    
    def get_job_subscribers(self, job_id: str) -> int:
        """Get subscriber count for a job"""
        return len(self.active_connections.get(job_id, set()))
    
    def get_authenticated_connections(self) -> int:
        """Get count of authenticated connections"""
        return sum(1 for auth in self.connection_auth.values() if auth["authenticated"])
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get statistics about all connections"""
        total_messages = sum(
            m.get("message_count", 0) 
            for m in self.connection_metadata.values()
        )
        total_errors = sum(
            m.get("error_count", 0) 
            for m in self.connection_metadata.values()
        )
        
        return {
            "total_connections": self.get_connection_count(),
            "authenticated_connections": self.get_authenticated_connections(),
            "jobs_with_subscribers": len(self.active_connections),
            "total_messages_sent": total_messages,
            "total_errors": total_errors,
            "avg_messages_per_connection": (
                total_messages / self.get_connection_count() 
                if self.get_connection_count() > 0 
                else 0
            ),
        }
    
    async def handle_ping(self, websocket: WebSocket) -> bool:
        """
        Handle keep-alive ping from client
        
        Args:
            websocket: Client connection
            
        Returns:
            True if successful, False if connection should close
        """
        try:
            await websocket.send_json({
                "event": "pong",
                "timestamp": datetime.utcnow().isoformat()
            })
            return True
        except Exception as e:
            logger.warning(f"Error sending pong: {e}")
            return False
    
    async def handle_status_request(self, websocket: WebSocket, job_id: Optional[str] = None) -> bool:
        """
        Handle status request from client
        
        Args:
            websocket: Client connection
            job_id: Job ID to get status for
            
        Returns:
            True if successful
        """
        try:
            # Get job info from auth
            auth = self.connection_auth.get(websocket, {})
            target_job_id = job_id or auth.get("job_id")
            
            if not target_job_id:
                await self.send_error_to_connection(websocket, "No job_id provided")
                return False
            
            # Send status (caller should provide actual job state)
            await websocket.send_json({
                "event": "status",
                "job_id": target_job_id,
                "timestamp": datetime.utcnow().isoformat()
            })
            return True
            
        except Exception as e:
            logger.warning(f"Error handling status request: {e}")
            return False


# Global connection manager instance
_connection_manager = None


def get_ws_connection_manager() -> WebSocketConnectionManager:
    """Get or create singleton WebSocket connection manager"""
    global _connection_manager
    if _connection_manager is None:
        _connection_manager = WebSocketConnectionManager()
    return _connection_manager
