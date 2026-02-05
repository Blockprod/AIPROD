"""
Tests for WebSocket Manager
Validates connection management, authentication, and event broadcasting
"""

import pytest
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from src.api.websocket_manager import WebSocketConnectionManager, get_ws_connection_manager


class MockWebSocket:
    """Mock WebSocket for testing"""
    
    def __init__(self):
        self.sent_messages = []
        self.is_open = True
    
    async def accept(self):
        self.is_open = True
    
    async def send_json(self, data):
        self.sent_messages.append(data)
    
    async def send_text(self, text):
        self.sent_messages.append(text)
    
    async def receive_text(self):
        return '{"msg": "test"}'
    
    async def close(self):
        self.is_open = False


class TestWebSocketConnectionManager:
    """Test suite for WebSocketConnectionManager"""
    
    @pytest.fixture
    def manager(self):
        """Create a fresh manager for each test"""
        return WebSocketConnectionManager()
    
    @pytest.mark.asyncio
    async def test_connect(self, manager):
        """Test connecting a new WebSocket"""
        ws = MockWebSocket()
        
        await manager.connect(ws, "job-123", "user-456")
        
        assert ws.is_open
        assert "job-123" in manager.active_connections
        assert ws in manager.active_connections["job-123"]
        assert manager.connection_auth[ws]["job_id"] == "job-123"
        assert manager.connection_auth[ws]["user_id"] == "user-456"
    
    @pytest.mark.asyncio
    async def test_connect_without_auth(self, manager):
        """Test connecting anonymous WebSocket"""
        ws = MockWebSocket()
        
        await manager.connect(ws, "job-123")
        
        assert manager.connection_auth[ws]["user_id"] is None
        assert manager.connection_auth[ws]["authenticated"] is False
    
    @pytest.mark.asyncio
    async def test_disconnect(self, manager):
        """Test disconnecting a WebSocket"""
        ws = MockWebSocket()
        await manager.connect(ws, "job-123", "user-456")
        
        await manager.disconnect(ws)
        
        assert ws not in manager.active_connections.get("job-123", set())
        assert ws not in manager.connection_auth
    
    @pytest.mark.asyncio
    async def test_multiple_connections_same_job(self, manager):
        """Test multiple clients subscribed to same job"""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        ws3 = MockWebSocket()
        
        await manager.connect(ws1, "job-123", "user-1")
        await manager.connect(ws2, "job-123", "user-2")
        await manager.connect(ws3, "job-123", "user-3")
        
        assert manager.get_connection_count("job-123") == 3
        assert manager.get_job_subscribers("job-123") == 3
    
    @pytest.mark.asyncio
    async def test_multiple_jobs(self, manager):
        """Test connections to different jobs"""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        
        await manager.connect(ws1, "job-123", "user-1")
        await manager.connect(ws2, "job-456", "user-1")
        
        assert manager.get_connection_count() == 2
        assert manager.get_job_subscribers("job-123") == 1
        assert manager.get_job_subscribers("job-456") == 1
    
    @pytest.mark.asyncio
    async def test_broadcast_to_job(self, manager):
        """Test broadcasting message to job subscribers"""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        
        await manager.connect(ws1, "job-123", "user-1")
        await manager.connect(ws2, "job-123", "user-2")
        
        message = {"event": "update", "state": "processing"}
        await manager.broadcast_to_job("job-123", message)
        
        # Both should receive the message
        assert len(ws1.sent_messages) >= 2  # connect + broadcast
        assert len(ws2.sent_messages) >= 2
        
        # Check message content
        last_ws1_msg = ws1.sent_messages[-1]
        assert last_ws1_msg["event"] == "update"
        assert last_ws1_msg["state"] == "processing"
    
    @pytest.mark.asyncio
    async def test_broadcast_only_to_target_job(self, manager):
        """Test that broadcast targets specific job only"""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        
        await manager.connect(ws1, "job-123", "user-1")
        await manager.connect(ws2, "job-456", "user-2")
        
        message = {"event": "update", "state": "done"}
        await manager.broadcast_to_job("job-123", message)
        
        # Only job-123 subscriber should get the message
        assert message in ws1.sent_messages or any(
            m.get("event") == "update" for m in ws1.sent_messages
        )
        # ws2 should only have connection message
        assert len(ws2.sent_messages) == 1
    
    @pytest.mark.asyncio
    async def test_broadcast_adds_timestamp(self, manager):
        """Test that broadcast adds timestamp to messages"""
        ws = MockWebSocket()
        await manager.connect(ws, "job-123")
        
        message = {"event": "update", "data": "test"}
        await manager.broadcast_to_job("job-123", message)
        
        # Find the update message
        update_msg = None
        for msg in ws.sent_messages:
            if msg.get("event") == "update":
                update_msg = msg
                break
        
        assert update_msg is not None
        assert "timestamp" in update_msg
    
    @pytest.mark.asyncio
    async def test_get_connection_count(self, manager):
        """Test connection counting"""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        ws3 = MockWebSocket()
        
        await manager.connect(ws1, "job-123")
        await manager.connect(ws2, "job-123")
        await manager.connect(ws3, "job-456")
        
        assert manager.get_connection_count() == 3
        assert manager.get_connection_count("job-123") == 2
        assert manager.get_connection_count("job-456") == 1
        assert manager.get_connection_count("job-999") == 0
    
    @pytest.mark.asyncio
    async def test_authenticated_connections_count(self, manager):
        """Test counting authenticated connections"""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        ws3 = MockWebSocket()
        
        await manager.connect(ws1, "job-123", "user-1")
        await manager.connect(ws2, "job-123", "user-2")
        await manager.connect(ws3, "job-123")  # Anonymous
        
        assert manager.get_authenticated_connections() == 2
    
    @pytest.mark.asyncio
    async def test_send_error_to_connection(self, manager):
        """Test sending error message to specific connection"""
        ws = MockWebSocket()
        await manager.connect(ws, "job-123")
        
        await manager.send_error_to_connection(ws, "Invalid command")
        
        # Find error message
        error_msg = None
        for msg in ws.sent_messages:
            if msg.get("event") == "error":
                error_msg = msg
                break
        
        assert error_msg is not None
        assert error_msg["error"] == "Invalid command"
    
    @pytest.mark.asyncio
    async def test_handle_ping(self, manager):
        """Test ping-pong keep-alive"""
        ws = MockWebSocket()
        await manager.connect(ws, "job-123")
        
        result = await manager.handle_ping(ws)
        
        assert result is True
        
        # Find pong message
        pong_msg = None
        for msg in ws.sent_messages:
            if msg.get("event") == "pong":
                pong_msg = msg
                break
        
        assert pong_msg is not None
    
    @pytest.mark.asyncio
    async def test_get_connection_stats(self, manager):
        """Test connection statistics"""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        
        await manager.connect(ws1, "job-123", "user-1")
        await manager.connect(ws2, "job-456", "user-2")
        
        await manager.broadcast_to_job("job-123", {"event": "test"})
        
        stats = manager.get_connection_stats()
        
        assert stats["total_connections"] == 2
        assert stats["authenticated_connections"] == 2
        assert stats["jobs_with_subscribers"] == 2
        assert "total_messages_sent" in stats
        assert "avg_messages_per_connection" in stats
    
    @pytest.mark.asyncio
    async def test_connection_cleanup_on_error(self, manager):
        """Test that broken connections are cleaned up"""
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        
        await manager.connect(ws1, "job-123")
        await manager.connect(ws2, "job-123")
        
        # Simulate ws1 failure by marking it as closed
        ws1.is_open = False
        
        await manager.disconnect(ws1)
        
        assert manager.get_connection_count("job-123") == 1
        assert ws1 not in manager.active_connections["job-123"]
        assert ws2 in manager.active_connections["job-123"]
    
    @pytest.mark.asyncio
    async def test_large_message_dropped(self, manager):
        """Test that oversized messages are dropped"""
        ws = MockWebSocket()
        manager.max_message_size = 100  # Very small limit for testing
        
        await manager.connect(ws, "job-123")
        
        # Create a message larger than limit
        large_msg = {"event": "data", "content": "x" * 500}
        await manager.broadcast_to_job("job-123", large_msg)
        
        # Should have connection message but not the large one
        assert len(ws.sent_messages) == 1  # Only connect message


class TestWebSocketSingleton:
    """Test WebSocket manager singleton pattern"""
    
    def test_get_ws_connection_manager_singleton(self):
        """Test that get_ws_connection_manager returns same instance"""
        # Reset the global
        import src.api.websocket_manager
        src.api.websocket_manager._connection_manager = None
        
        manager1 = get_ws_connection_manager()
        manager2 = get_ws_connection_manager()
        
        assert manager1 is manager2


class TestWebSocketEndpoints:
    """Test WebSocket endpoints exist"""
    
    def test_websocket_endpoint_exists(self):
        """Test that WebSocket endpoint is registered"""
        from src.api.main import app
        
        # Look for WebSocket route (filter for routes with path attribute)
        ws_routes = [getattr(r, 'path', '') for r in app.routes if 'ws' in getattr(r, 'path', '').lower()]
        
        # Should have at least the job updates endpoint
        assert any("job" in route for route in ws_routes) or len(ws_routes) > 0
