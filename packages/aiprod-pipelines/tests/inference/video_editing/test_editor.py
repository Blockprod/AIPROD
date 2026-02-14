"""Tests for video editing system"""

import pytest
import tempfile
import numpy as np
from pathlib import Path

from aiprod_pipelines.inference.video_editing import (
    VideoEditorBackend,
    EditorState,
    EditOperation,
    EditResponse,
)


class TestVideoEditorBackend:
    """Test video editor backend"""
    
    @pytest.fixture
    def editor(self):
        """Create editor instance"""
        return VideoEditorBackend(device="cpu")
    
    def test_backend_initialization(self, editor):
        """Test backend creates without errors"""
        assert editor is not None
        assert editor.device == "cpu"
        assert len(editor.active_sessions) == 0
    
    def test_editor_state_creation(self):
        """Test editor state"""
        state = EditorState(
            video_path="/test/video.mp4",
            current_frame=0,
            total_frames=120,
            fps=30.0,
            resolution=(1920, 1080),
        )
        
        assert state.video_path == "/test/video.mp4"
        assert state.total_frames == 120
        assert state.fps == 30.0
        assert state.resolution == (1920, 1080)
        assert state.current_frame == 0
    
    def test_edit_operation_creation(self):
        """Test edit operation"""
        op = EditOperation(
            operation_id="op_001",
            operation_type="adjust_brightness",
            start_frame=0,
            end_frame=30,
            parameters={"value": 10},
        )
        
        assert op.operation_id == "op_001"
        assert op.operation_type == "adjust_brightness"
        assert op.start_frame == 0
        assert op.end_frame == 30
        assert op.parameters["value"] == 10
    
    def test_edit_response_creation(self):
        """Test edit response"""
        from aiprod_pipelines.inference.video_editing import EditResponse
        
        response = EditResponse(
            success=True,
            message="Operation successful",
            operation_id="op_001",
        )
        
        assert response.success is True
        assert response.message == "Operation successful"
        assert response.operation_id == "op_001"
    
    def test_undo_stack_initialization(self):
        """Test undo stack"""
        state = EditorState(
            video_path="/test.mp4",
            current_frame=0,
            total_frames=100,
            fps=30.0,
            resolution=(1920, 1080),
        )
        
        assert len(state.operations_history) == 0
        assert len(state.undo_stack) == 0
    
    def test_frame_cache_management(self, editor):
        """Test frame cache"""
        # Initially empty
        assert len(editor.frame_cache) == 0
        
        # Add frame to cache
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        editor.frame_cache["test.mp4:0"] = frame
        
        assert len(editor.frame_cache) == 1
        assert editor.frame_cache["test.mp4:0"].shape == (1080, 1920, 3)


class TestEditOperations:
    """Test individual edit operations"""
    
    @pytest.mark.asyncio
    async def test_edit_operation_basic(self):
        """Test basic edit operation"""
        editor = VideoEditorBackend()
        
        # Create simple frame
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        
        # Create operation
        op = EditOperation(
            operation_id="test_op",
            operation_type="adjust_brightness",
            start_frame=0,
            end_frame=10,
            parameters={"value": 20},
        )
        
        # Apply operation
        result = await editor._apply_operation(frame, op)
        
        assert result is not None
        assert result.shape == frame.shape


class TestAPIGateway:
    """Test API gateway"""
    
    def test_api_gateway_initialization(self):
        """Test gateway creates FastAPI app"""
        from aiprod_pipelines.inference.video_editing import APIGateway
        
        gateway = APIGateway()
        app = gateway.get_app()
        
        assert app is not None
        # Check routes are registered
        routes = [r.path for r in app.routes]
        assert any("/api/editor" in r for r in routes)


class TestVideoEditorWorkflow:
    """Test complete editor workflow"""
    
    def test_editor_workflow_initialization(self):
        """Test workflow initialization"""
        editor = VideoEditorBackend()
        
        state = EditorState(
            video_path="/videos/test.mp4",
            current_frame=0,
            total_frames=300,
            fps=24.0,
            resolution=(1920, 1080),
        )
        
        # Simulate session
        editor.active_sessions["session_001"] = state
        
        assert "session_001" in editor.active_sessions
        assert editor.active_sessions["session_001"].total_frames == 300
    
    @pytest.mark.asyncio
    async def test_undo_redo_workflow(self):
        """Test undo/redo workflow"""
        editor = VideoEditorBackend()
        
        state = EditorState(
            video_path="/test.mp4",
            current_frame=0,
            total_frames=100,
            fps=30.0,
            resolution=(1920, 1080),
        )
        
        session_id = "test_session"
        editor.active_sessions[session_id] = state
        
        # Create operation
        op = EditOperation(
            operation_id="op_1",
            operation_type="adjust_brightness",
            start_frame=0,
            end_frame=30,
            parameters={"value": 10},
        )
        state.operations_history.append(op)
        
        # Undo
        response = await editor.undo(session_id)
        assert response.success is True
        assert len(state.operations_history) == 0
        assert len(state.undo_stack) == 1
        
        # Redo
        response = await editor.redo(session_id)
        assert response.success is True
        assert len(state.operations_history) == 1
        assert len(state.undo_stack) == 0
    
    @pytest.mark.asyncio
    async def test_editing_summary(self):
        """Test editing summary generation"""
        editor = VideoEditorBackend()
        
        state = EditorState(
            video_path="/test.mp4",
            current_frame=0,
            total_frames=100,
            fps=30.0,
            resolution=(1920, 1080),
        )
        
        # Add operations
        op1 = EditOperation(
            operation_id="op_1",
            operation_type="adjust_brightness",
            start_frame=0,
            end_frame=30,
            parameters={"value": 10},
        )
        
        op2 = EditOperation(
            operation_id="op_2",
            operation_type="adjust_contrast",
            start_frame=30,
            end_frame=60,
            parameters={"value": 1.5},
        )
        
        state.operations_history.append(op1)
        state.operations_history.append(op2)
        
        session_id = "test"
        editor.active_sessions[session_id] = state
        
        summary = await editor.get_editing_summary(session_id)
        
        assert summary["total_operations"] == 2
        assert "adjust_brightness" in summary["operations_by_type"]
        assert "adjust_contrast" in summary["operations_by_type"]
        assert len(summary["operations_by_type"]["adjust_brightness"]) == 1
        assert len(summary["operations_by_type"]["adjust_contrast"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
