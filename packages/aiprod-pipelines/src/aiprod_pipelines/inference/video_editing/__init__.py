"""Video Editing UI System

Real-time interactive video editing with:
- Frame scrubber and timeline
- Edit operations (brightness, contrast, blur, etc)
- Undo/redo stack (50+ actions)
- GPU-accelerated rendering
- <200ms frame navigation
- Export to MP4
"""

from .backend import (
    VideoEditorBackend,
    EditorState,
    EditOperation,
    EditResponse,
)
from .api_gateway import APIGateway

__all__ = [
    "VideoEditorBackend",
    "EditorState",
    "EditOperation",
    "EditResponse",
    "APIGateway",
]
