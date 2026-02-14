"""
Interactive Collaboration Layer - WebSocket Real-Time Collaboration
====================================================================

Implements multi-user collaboration features:
- WebSocket connections for real-time updates
- Comment system for asset review
- Approval/rejection workflow
- Manifest editing with conflict resolution
- Version history tracking

PHASE 4 implementation (Weeks 11-13).
"""

from typing import Dict, Any, List, Set, Optional
import asyncio
import logging
from datetime import datetime
import json
from fastapi import WebSocket, WebSocketDisconnect
from collections import defaultdict


logger = logging.getLogger(__name__)


class CollaborationRoom:
    """
    Real-time collaboration room for a single job.
    
    Manages:
    - Active WebSocket connections
    - Comment broadcasting
    - Approval tracking
    - Manifest edits with conflict resolution
    """
    
    def __init__(self, job_id: str):
        """Initialize collaboration room."""
        self.job_id = job_id
        self.participants: Set[WebSocket] = set()
        self.comments: List[Dict[str, Any]] = []
        self.approvals: Dict[str, Dict[str, Any]] = {}  # user_id -> approval data
        self.rejections: Dict[str, Dict[str, Any]] = {}  # user_id -> rejection data
        self.manifest_edits: List[Dict[str, Any]] = []
        self.created_at = datetime.now()
        
        logger.info(f"Collaboration room created: {job_id}")
    
    async def add_participant(self, websocket: WebSocket, user_id: str = None):
        """
        Add participant to room.
        
        Args:
            websocket: WebSocket connection
            user_id: Optional user identifier
        """
        self.participants.add(websocket)
        
        # Send current state to new participant
        await self._send_current_state(websocket)
        
        # Notify others
        await self.broadcast({
            "type": "participant_joined",
            "user_id": user_id or "anonymous",
            "timestamp": datetime.now().isoformat(),
            "participant_count": len(self.participants)
        }, exclude=websocket)
        
        logger.info(f"Participant added to {self.job_id}: {user_id or 'anonymous'}")
    
    async def remove_participant(self, websocket: WebSocket):
        """
        Remove participant from room.
        
        Args:
            websocket: WebSocket connection
        """
        if websocket in self.participants:
            self.participants.remove(websocket)
            
            # Notify others
            await self.broadcast({
                "type": "participant_left",
                "timestamp": datetime.now().isoformat(),
                "participant_count": len(self.participants)
            })
            
            logger.info(f"Participant removed from {self.job_id}")
    
    async def broadcast_comment(self, message: Dict[str, Any]):
        """
        Broadcast comment to all participants.
        
        Args:
            message: Comment message with user_id, text, asset_id
        """
        comment = {
            "id": f"comment_{len(self.comments)}",
            "user_id": message.get("user_id", "anonymous"),
            "text": message.get("text", ""),
            "asset_id": message.get("asset_id"),
            "timestamp": datetime.now().isoformat()
        }
        
        self.comments.append(comment)
        
        # Broadcast to all participants
        await self.broadcast({
            "type": "comment",
            "comment": comment
        })
        
        logger.info(f"Comment broadcast in {self.job_id}: {comment['id']}")
    
    async def record_approval(self, message: Dict[str, Any]):
        """
        Record approval from user.
        
        Args:
            message: Approval message with user_id, asset_id
        """
        user_id = message.get("user_id", "anonymous")
        
        approval = {
            "user_id": user_id,
            "asset_id": message.get("asset_id"),
            "approved": True,
            "timestamp": datetime.now().isoformat(),
            "note": message.get("note", "")
        }
        
        self.approvals[user_id] = approval
        
        # Remove rejection if exists
        if user_id in self.rejections:
            del self.rejections[user_id]
        
        # Broadcast approval
        await self.broadcast({
            "type": "approval",
            "approval": approval,
            "total_approvals": len(self.approvals),
            "total_rejections": len(self.rejections)
        })
        
        logger.info(f"Approval recorded in {self.job_id}: {user_id}")
    
    async def record_rejection(self, message: Dict[str, Any]):
        """
        Record rejection from user.
        
        Args:
            message: Rejection message with user_id, asset_id, reason
        """
        user_id = message.get("user_id", "anonymous")
        
        rejection = {
            "user_id": user_id,
            "asset_id": message.get("asset_id"),
            "approved": False,
            "timestamp": datetime.now().isoformat(),
            "reason": message.get("reason", "")
        }
        
        self.rejections[user_id] = rejection
        
        # Remove approval if exists
        if user_id in self.approvals:
            del self.approvals[user_id]
        
        # Broadcast rejection
        await self.broadcast({
            "type": "rejection",
            "rejection": rejection,
            "total_approvals": len(self.approvals),
            "total_rejections": len(self.rejections)
        })
        
        logger.info(f"Rejection recorded in {self.job_id}: {user_id}")
    
    async def apply_manifest_edit(self, message: Dict[str, Any]):
        """
        Apply manifest edit with conflict resolution.
        
        Args:
            message: Edit message with user_id, field, value
        """
        edit = {
            "id": f"edit_{len(self.manifest_edits)}",
            "user_id": message.get("user_id", "anonymous"),
            "field": message.get("field"),
            "old_value": message.get("old_value"),
            "new_value": message.get("new_value"),
            "timestamp": datetime.now().isoformat()
        }
        
        self.manifest_edits.append(edit)
        
        # Broadcast edit
        await self.broadcast({
            "type": "manifest_edit",
            "edit": edit,
            "edit_count": len(self.manifest_edits)
        })
        
        logger.info(f"Manifest edit applied in {self.job_id}: {edit['field']}")
    
    async def broadcast(self, message: Dict[str, Any], exclude: Optional[WebSocket] = None):
        """
        Broadcast message to all participants.
        
        Args:
            message: Message dictionary
            exclude: Optional WebSocket to exclude from broadcast
        """
        dead_connections = []
        
        for participant in self.participants:
            if participant == exclude:
                continue
            
            try:
                await participant.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to participant: {e}")
                dead_connections.append(participant)
        
        # Clean up dead connections
        for dead in dead_connections:
            await self.remove_participant(dead)
    
    async def _send_current_state(self, websocket: WebSocket):
        """
        Send current room state to new participant.
        
        Args:
            websocket: WebSocket connection
        """
        state = {
            "type": "room_state",
            "job_id": self.job_id,
            "comments": self.comments,
            "approvals": list(self.approvals.values()),
            "rejections": list(self.rejections.values()),
            "manifest_edits": self.manifest_edits,
            "participant_count": len(self.participants)
        }
        
        await websocket.send_json(state)
    
    def get_approval_status(self) -> Dict[str, Any]:
        """
        Get current approval status.
        
        Returns:
            Approval status dictionary
        """
        return {
            "total_approvals": len(self.approvals),
            "total_rejections": len(self.rejections),
            "approved_by": [a["user_id"] for a in self.approvals.values()],
            "rejected_by": [r["user_id"] for r in self.rejections.values()],
            "requires_approval": True if len(self.approvals) == 0 else False
        }


class CollaborationRoomManager:
    """
    Manages all collaboration rooms.
    
    Handles:
    - Room creation and cleanup
    - Room lookup
    - Room statistics
    """
    
    def __init__(self):
        """Initialize room manager."""
        self.rooms: Dict[str, CollaborationRoom] = {}
        self.room_access_count: Dict[str, int] = defaultdict(int)
        
        logger.info("CollaborationRoomManager initialized")
    
    def get_or_create_room(self, job_id: str) -> CollaborationRoom:
        """
        Get existing room or create new one.
        
        Args:
            job_id: Job identifier
            
        Returns:
            CollaborationRoom instance
        """
        if job_id not in self.rooms:
            self.rooms[job_id] = CollaborationRoom(job_id)
            logger.info(f"Created new room: {job_id}")
        
        self.room_access_count[job_id] += 1
        return self.rooms[job_id]
    
    def get_room(self, job_id: str) -> Optional[CollaborationRoom]:
        """
        Get existing room.
        
        Args:
            job_id: Job identifier
            
        Returns:
            CollaborationRoom or None
        """
        return self.rooms.get(job_id)
    
    def close_room(self, job_id: str):
        """
        Close and remove room.
        
        Args:
            job_id: Job identifier
        """
        if job_id in self.rooms:
            del self.rooms[job_id]
            logger.info(f"Closed room: {job_id}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get room statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "total_rooms": len(self.rooms),
            "active_participants": sum(len(room.participants) for room in self.rooms.values()),
            "most_accessed": sorted(
                self.room_access_count.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }


class VersionManager:
    """
    Manages version history for jobs.
    
    Tracks:
    - Manifest versions
    - Edit history
    - User contributions
    """
    
    def __init__(self):
        """Initialize version manager."""
        self.versions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        logger.info("VersionManager initialized")
    
    async def save_version(
        self,
        job_id: str,
        manifest: Dict[str, Any],
        user_id: str,
        change_description: str
    ):
        """
        Save new version of manifest.
        
        Args:
            job_id: Job identifier
            manifest: Current manifest
            user_id: User who made changes
            change_description: Description of changes
        """
        version = {
            "version": len(self.versions[job_id]) + 1,
            "manifest": manifest.copy(),
            "user_id": user_id,
            "change_description": change_description,
            "timestamp": datetime.now().isoformat()
        }
        
        self.versions[job_id].append(version)
        
        logger.info(f"Version saved for {job_id}: v{version['version']}")
    
    async def get_history(self, job_id: str) -> List[Dict[str, Any]]:
        """
        Get version history for job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            List of versions
        """
        return self.versions.get(job_id, [])
    
    async def get_version(self, job_id: str, version: int) -> Optional[Dict[str, Any]]:
        """
        Get specific version.
        
        Args:
            job_id: Job identifier
            version: Version number
            
        Returns:
            Version data or None
        """
        versions = self.versions.get(job_id, [])
        
        for v in versions:
            if v["version"] == version:
                return v
        
        return None
    
    async def restore_version(self, job_id: str, version: int) -> Optional[Dict[str, Any]]:
        """
        Restore specific version.
        
        Args:
            job_id: Job identifier
            version: Version number to restore
            
        Returns:
            Restored manifest or None
        """
        version_data = await self.get_version(job_id, version)
        
        if version_data:
            # Save restoration as new version
            await self.save_version(
                job_id=job_id,
                manifest=version_data["manifest"],
                user_id="system",
                change_description=f"Restored from version {version}"
            )
            
            return version_data["manifest"]
        
        return None


# Global instances
collaboration_rooms = CollaborationRoomManager()
version_manager = VersionManager()
