"""
Recording management API endpoints.

In-memory mock implementation: lets the UI "Recordings" tab render and operate
(list / start / stop / delete) without a real CSI capture backend. Replace the
in-memory store with a real recording service when available.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


class RecordingInfo(BaseModel):
    """Description of a single CSI recording."""

    id: str = Field(..., description="Unique recording identifier")
    name: str = Field(..., description="Recording name/label")
    status: str = Field(default="completed", description="recording | completed")
    created_at: str = Field(..., description="ISO timestamp")
    duration_seconds: float = Field(default=0.0)
    frames: int = Field(default=0)


class RecordingListResponse(BaseModel):
    """Response model for listing recordings."""

    recordings: List[RecordingInfo] = Field(default_factory=list)
    count: int = Field(default=0)


# In-memory mock store
_RECORDINGS: Dict[str, RecordingInfo] = {}
_CURRENT: Dict[str, Optional[str]] = {"id": None}


@router.get("/list", response_model=RecordingListResponse)
async def list_recordings():
    """List all recordings."""
    recordings = list(_RECORDINGS.values())
    return RecordingListResponse(recordings=recordings, count=len(recordings))


@router.post("/start")
async def start_recording(config: Optional[Dict[str, Any]] = None):
    """Start a new CSI recording."""
    rec_id = uuid.uuid4().hex[:12]
    name = (config or {}).get("name") or f"recording_{rec_id}"
    rec = RecordingInfo(
        id=rec_id,
        name=name,
        status="recording",
        created_at=datetime.now().isoformat(),
    )
    _RECORDINGS[rec_id] = rec
    _CURRENT["id"] = rec_id
    logger.info("Recording started (mock): %s", rec_id)
    return {"status": "started", "recording": rec}


@router.post("/stop")
async def stop_recording():
    """Stop the current recording."""
    rec_id = _CURRENT["id"]
    _CURRENT["id"] = None
    if rec_id and rec_id in _RECORDINGS:
        _RECORDINGS[rec_id].status = "completed"
        logger.info("Recording stopped (mock): %s", rec_id)
        return {"status": "stopped", "recording": _RECORDINGS[rec_id]}
    return {"status": "stopped", "recording": None}


@router.delete("/{recording_id}")
async def delete_recording(recording_id: str):
    """Delete a recording."""
    if recording_id not in _RECORDINGS:
        raise HTTPException(status_code=404, detail=f"Recording '{recording_id}' not found")
    del _RECORDINGS[recording_id]
    if _CURRENT["id"] == recording_id:
        _CURRENT["id"] = None
    logger.info("Recording deleted (mock): %s", recording_id)
    return {"status": "deleted", "recording_id": recording_id}
