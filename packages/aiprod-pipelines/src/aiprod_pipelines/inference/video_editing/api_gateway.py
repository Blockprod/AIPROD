"""API Gateway for video editing operations"""

from fastapi import FastAPI, WebSocket, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import Optional, Dict, List
import uuid
import json

from .backend import VideoEditorBackend, EditResponse


# Request/Response Models
class OpenVideoRequest(BaseModel):
    video_path: str


class EditOperationRequest(BaseModel):
    operation_type: str
    start_frame: int
    end_frame: int
    parameters: Dict = {}


class ExportVideoRequest(BaseModel):
    output_path: str
    fps: Optional[float] = None


class FrameResponse(BaseModel):
    session_id: str
    frame_index: int
    width: int
    height: int
    base64_data: str  # Base64 encoded frame


class EditorStateResponse(BaseModel):
    session_id: str
    video_path: str
    current_frame: int
    total_frames: int
    fps: float
    resolution: tuple
    operations_count: int
    undo_available: bool
    redo_available: bool


class APIGateway:
    """REST API gateway for video editor"""
    
    def __init__(self):
        """Initialize API gateway"""
        self.backend = VideoEditorBackend()
        self.app = FastAPI(title="Video Editor API")
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup all API routes"""
        
        @self.app.post("/api/editor/open")
        async def open_video(request: OpenVideoRequest):
            """Open video for editing"""
            try:
                session_id = str(uuid.uuid4())[:8]
                state = await self.backend.open_video(session_id, request.video_path)
                
                return {
                    "success": True,
                    "session_id": session_id,
                    "editor_state": {
                        "video_path": state.video_path,
                        "current_frame": state.current_frame,
                        "total_frames": state.total_frames,
                        "fps": state.fps,
                        "resolution": state.resolution,
                    },
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/api/editor/{session_id}/frame/{frame_index}")
        async def get_frame(session_id: str, frame_index: int):
            """Get single frame with edits applied"""
            try:
                import base64
                import cv2
                
                frame = await self.backend.get_frame(session_id, frame_index)
                
                # Encode to JPEG for transmission
                success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                if not success:
                    raise RuntimeError("Failed to encode frame")
                
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                return {
                    "success": True,
                    "frame_index": frame_index,
                    "data": frame_base64,
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/api/editor/{session_id}/edit")
        async def create_edit(session_id: str, request: EditOperationRequest):
            """Create edit operation"""
            try:
                response = await self.backend.create_edit_operation(
                    session_id=session_id,
                    operation_type=request.operation_type,
                    start_frame=request.start_frame,
                    end_frame=request.end_frame,
                    **request.parameters,
                )
                
                if not response.success:
                    raise HTTPException(status_code=400, detail=response.message)
                
                import base64
                import cv2
                
                frame_base64 = None
                if response.edited_frame is not None:
                    success, buffer = cv2.imencode('.jpg', response.edited_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if success:
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                return {
                    "success": True,
                    "message": response.message,
                    "operation_id": response.operation_id,
                    "preview": frame_base64,
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/api/editor/{session_id}/undo")
        async def undo_operation(session_id: str):
            """Undo last operation"""
            try:
                response = await self.backend.undo(session_id)
                
                if not response.success:
                    raise HTTPException(status_code=400, detail=response.message)
                
                return {
                    "success": True,
                    "message": response.message,
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/api/editor/{session_id}/redo")
        async def redo_operation(session_id: str):
            """Redo last undone operation"""
            try:
                response = await self.backend.redo(session_id)
                
                if not response.success:
                    raise HTTPException(status_code=400, detail=response.message)
                
                return {
                    "success": True,
                    "message": response.message,
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/api/editor/{session_id}/export")
        async def export_video(session_id: str, request: ExportVideoRequest):
            """Export edited video"""
            try:
                response = await self.backend.export_video(
                    session_id=session_id,
                    output_path=request.output_path,
                    fps=request.fps,
                )
                
                if not response.success:
                    raise HTTPException(status_code=400, detail=response.message)
                
                return {
                    "success": True,
                    "message": response.message,
                }
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/api/editor/{session_id}/summary")
        async def get_summary(session_id: str):
            """Get editing summary"""
            try:
                summary = await self.backend.get_editing_summary(session_id)
                
                return {
                    "success": True,
                    "summary": summary,
                }
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
    
    def get_app(self) -> FastAPI:
        """Get FastAPI app instance"""
        return self.app
