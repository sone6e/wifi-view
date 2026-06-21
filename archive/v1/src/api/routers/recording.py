"""
Recording management API endpoints.

Uses the UDP aggregator bridge to capture real CSI frames from ESP32 nodes.
Falls back to mock behavior if the aggregator is not receiving data.
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.bridge.udp_aggregator import get_aggregator
from src.bridge.csi_recorder import CSIRecorder

logger = logging.getLogger(__name__)
router = APIRouter()

# Shared recorder instance
_recorder = CSIRecorder()


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


# In-memory state for active recording
_CURRENT: Dict[str, Optional[str]] = {"id": None}


@router.get("/list", response_model=RecordingListResponse)
async def list_recordings():
    """List all recordings (from disk)."""
    disk_recordings = _recorder.list_recordings()
    recordings = [
        RecordingInfo(
            id=r["id"],
            name=r.get("label", r["id"]),
            status="completed",
            created_at="",
            duration_seconds=r.get("duration_s", 0),
            frames=r.get("frames", 0),
        )
        for r in disk_recordings
    ]
    return RecordingListResponse(recordings=recordings, count=len(recordings))


@router.post("/start")
async def start_recording(config: Optional[Dict[str, Any]] = None):
    """Start a new CSI recording from ESP32 data."""
    label = (config or {}).get("name") or (config or {}).get("label") or "unlabeled"

    # Start the recorder
    rec_id = _recorder.start(label=label)
    _CURRENT["id"] = rec_id

    # Connect recorder to the UDP aggregator if available
    aggregator = get_aggregator()
    if aggregator and aggregator.is_receiving:
        aggregator.on_frame(_recorder.add_frame)
        source = "esp32_live"
    else:
        source = "waiting_for_esp32"

    rec = RecordingInfo(
        id=rec_id,
        name=label,
        status="recording",
        created_at=datetime.now().isoformat(),
    )
    logger.info(f"Recording started: {rec_id} (source={source})")
    return {"status": "started", "recording": rec, "source": source}


@router.post("/stop")
async def stop_recording():
    """Stop the current recording and save to disk."""
    _CURRENT["id"] = None
    result = _recorder.stop()
    rec = RecordingInfo(
        id=result.get("id", ""),
        name=result.get("label", ""),
        status=result.get("status", "completed"),
        created_at="",
        duration_seconds=result.get("duration_s", 0),
        frames=result.get("frames", 0),
    )
    logger.info(f"Recording stopped: {result}")
    return {"status": "stopped", "recording": rec}


@router.delete("/{recording_id}")
async def delete_recording(recording_id: str):
    """Delete a recording from disk."""
    success = _recorder.delete_recording(recording_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Recording '{recording_id}' not found")
    if _CURRENT["id"] == recording_id:
        _CURRENT["id"] = None
    logger.info(f"Recording deleted: {recording_id}")
    return {"status": "deleted", "recording_id": recording_id}
