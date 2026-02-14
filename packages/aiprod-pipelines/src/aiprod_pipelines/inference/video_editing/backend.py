"""
Real-time Video Editing Backend System

Supports interactive frame-level video editing:
- Frame scrubber with preview
- Timeline controls
- Undo/redo functionality
- Real-time GPU rendering
- Frame-level editing operations
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import asyncio
import numpy as np


@dataclass
class EditOperation:
    """Single edit operation on video"""
    operation_id: str
    operation_type: str  # "trim", "cut", "adjust_brightness", "adjust_contrast", "add_filter"
    start_frame: int
    end_frame: int
    parameters: Dict = field(default_factory=dict)
    timestamp: float = 0.0
    
    def __hash__(self):
        return hash(self.operation_id)


@dataclass
class EditorState:
    """Current state of editor session"""
    video_path: str
    current_frame: int
    total_frames: int
    fps: float
    resolution: Tuple[int, int]  # (width, height)
    
    operations_history: List[EditOperation] = field(default_factory=list)
    undo_stack: List[EditOperation] = field(default_factory=list)  # Redo stack
    
    playback_speed: float = 1.0
    is_playing: bool = False
    selection_start: Optional[int] = None
    selection_end: Optional[int] = None


@dataclass
class EditResponse:
    """Response after edit operation"""
    success: bool
    message: str
    edited_frame: Optional[np.ndarray] = None  # Preview frame
    operation_id: Optional[str] = None


class VideoEditorBackend:
    """Backend for real-time video editing"""
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize editor backend.
        
        Args:
            device: "cuda" or "cpu"
        """
        self.device = device
        self.active_sessions: Dict[str, EditorState] = {}
        self.frame_cache: Dict[str, np.ndarray] = {}
        self.max_cache_size = 100
    
    async def open_video(
        self,
        session_id: str,
        video_path: str,
    ) -> EditorState:
        """
        Open video for editing.
        
        Args:
            session_id: Unique session identifier
            video_path: Path to video file
            
        Returns:
            Initial editor state
        """
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        state = EditorState(
            video_path=video_path,
            current_frame=0,
            total_frames=total_frames,
            fps=fps,
            resolution=(width, height),
        )
        
        self.active_sessions[session_id] = state
        return state
    
    async def get_frame(
        self,
        session_id: str,
        frame_index: int,
    ) -> np.ndarray:
        """Get frame at index with all edits applied"""
        if session_id not in self.active_sessions:
            raise ValueError(f"Invalid session: {session_id}")
        
        state = self.active_sessions[session_id]
        
        cache_key = f"{state.video_path}:{frame_index}"
        if cache_key in self.frame_cache:
            return self.frame_cache[cache_key].copy()
        
        import cv2
        
        cap = cv2.VideoCapture(state.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise RuntimeError(f"Failed to load frame {frame_index}")
        
        # Apply all edits
        for op in state.operations_history:
            if op.start_frame <= frame_index <= op.end_frame:
                frame = await self._apply_operation(frame, op)
        
        # Cache frame
        if len(self.frame_cache) > self.max_cache_size:
            oldest_key = next(iter(self.frame_cache))
            del self.frame_cache[oldest_key]
        
        self.frame_cache[cache_key] = frame.copy()
        
        return frame
    
    async def _apply_operation(self, frame: np.ndarray, operation: EditOperation) -> np.ndarray:
        """Apply single edit operation to frame"""
        import cv2
        
        if operation.operation_type == "adjust_brightness":
            brightness = operation.parameters.get("value", 0)
            frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=brightness)
        
        elif operation.operation_type == "adjust_contrast":
            contrast = operation.parameters.get("value", 1.0)
            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=0)
        
        elif operation.operation_type == "adjust_saturation":
            saturation = operation.parameters.get("value", 1.0)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = hsv[:, :, 1] * saturation
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        elif operation.operation_type == "blur":
            blur_amount = operation.parameters.get("amount", 5)
            frame = cv2.blur(frame, (blur_amount, blur_amount))
        
        elif operation.operation_type == "sharpen":
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]]) / 1.0
            frame = cv2.filter2D(frame, -1, kernel)
        
        return frame
    
    async def create_edit_operation(
        self,
        session_id: str,
        operation_type: str,
        start_frame: int,
        end_frame: int,
        **parameters
    ) -> EditResponse:
        """
        Create new edit operation.
        
        Args:
            session_id: Session ID
            operation_type: Type of operation
            start_frame: Starting frame
            end_frame: Ending frame (inclusive)
            **parameters: Operation-specific parameters
            
        Returns:
            EditResponse with result
        """
        if session_id not in self.active_sessions:
            return EditResponse(False, "Invalid session")
        
        import uuid
        import time
        
        state = self.active_sessions[session_id]
        
        # Validate frames
        if start_frame < 0 or end_frame >= state.total_frames or start_frame > end_frame:
            return EditResponse(False, "Invalid frame range")
        
        # Create operation
        operation_id = str(uuid.uuid4())[:8]
        operation = EditOperation(
            operation_id=operation_id,
            operation_type=operation_type,
            start_frame=start_frame,
            end_frame=end_frame,
            parameters=parameters,
            timestamp=time.time(),
        )
        
        # Add to history
        state.operations_history.append(operation)
        state.undo_stack.clear()  # Clear redo stack on new operation
        
        # Get preview frame
        try:
            preview_frame = await self.get_frame(session_id, start_frame)
            return EditResponse(
                True,
                f"Operation created: {operation_type}",
                edited_frame=preview_frame,
                operation_id=operation_id,
            )
        except Exception as e:
            return EditResponse(False, f"Error: {str(e)}")
    
    async def undo(self, session_id: str) -> EditResponse:
        """Undo last operation"""
        if session_id not in self.active_sessions:
            return EditResponse(False, "Invalid session")
        
        state = self.active_sessions[session_id]
        
        if not state.operations_history:
            return EditResponse(False, "Nothing to undo")
        
        # Move operation to undo stack
        operation = state.operations_history.pop()
        state.undo_stack.append(operation)
        
        # Clear cache for affected frames
        self._clear_cache_range(state.video_path, operation.start_frame, operation.end_frame)
        
        return EditResponse(True, f"Undone: {operation.operation_type}")
    
    async def redo(self, session_id: str) -> EditResponse:
        """Redo last undone operation"""
        if session_id not in self.active_sessions:
            return EditResponse(False, "Invalid session")
        
        state = self.active_sessions[session_id]
        
        if not state.undo_stack:
            return EditResponse(False, "Nothing to redo")
        
        # Move operation back to history
        operation = state.undo_stack.pop()
        state.operations_history.append(operation)
        
        # Clear cache
        self._clear_cache_range(state.video_path, operation.start_frame, operation.end_frame)
        
        return EditResponse(True, f"Redone: {operation.operation_type}")
    
    def _clear_cache_range(self, video_path: str, start_frame: int, end_frame: int):
        """Clear cache for frame range"""
        keys_to_delete = [
            k for k in self.frame_cache.keys()
            if k.startswith(f"{video_path}:") and 
            start_frame <= int(k.split(":")[1]) <= end_frame
        ]
        for k in keys_to_delete:
            del self.frame_cache[k]
    
    async def export_video(
        self,
        session_id: str,
        output_path: str,
        fps: Optional[float] = None,
    ) -> EditResponse:
        """
        Export edited video.
        
        Args:
            session_id: Session ID
            output_path: Output file path
            fps: FPS for output (default: same as input)
            
        Returns:
            EditResponse with result
        """
        if session_id not in self.active_sessions:
            return EditResponse(False, "Invalid session")
        
        state = self.active_sessions[session_id]
        
        if fps is None:
            fps = state.fps
        
        import cv2
        
        try:
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                state.resolution,
            )
            
            # Write all frames with edits applied
            for frame_idx in range(state.total_frames):
                if frame_idx % 30 == 0:
                    print(f"  Exporting... {frame_idx}/{state.total_frames}")
                
                frame = await self.get_frame(session_id, frame_idx)
                writer.write(frame)
            
            writer.release()
            
            return EditResponse(True, f"Video exported to {output_path}")
        
        except Exception as e:
            return EditResponse(False, f"Export failed: {str(e)}")
    
    async def get_editing_summary(self, session_id: str) -> Dict:
        """Get summary of all edits"""
        if session_id not in self.active_sessions:
            return {}
        
        state = self.active_sessions[session_id]
        
        # Group operations by type
        operations_by_type = {}
        for op in state.operations_history:
            key = op.operation_type
            if key not in operations_by_type:
                operations_by_type[key] = []
            operations_by_type[key].append({
                "id": op.operation_id,
                "frames": (op.start_frame, op.end_frame),
                "parameters": op.parameters,
            })
        
        return {
            "total_operations": len(state.operations_history),
            "operations_by_type": operations_by_type,
            "undo_stack_size": len(state.undo_stack),
            "video_info": {
                "total_frames": state.total_frames,
                "fps": state.fps,
                "resolution": state.resolution,
            },
        }
