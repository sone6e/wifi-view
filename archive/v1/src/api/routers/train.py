"""
Training management API endpoints.

In-memory mock implementation: lets the UI "Training" tab render and operate
(start / stop / status / pretrain / lora) without a real training pipeline.
Replace the in-memory state with a real training service when available.
"""

import logging
from typing import Any, Dict, Optional
from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


class TrainingStatus(BaseModel):
    """Current training status."""

    status: str = Field(default="idle", description="idle | running | completed | error")
    progress: float = Field(default=0.0, description="Progress 0.0 - 1.0")
    epoch: int = Field(default=0)
    total_epochs: int = Field(default=0)
    message: Optional[str] = Field(default=None)
    started_at: Optional[str] = Field(default=None)


# In-memory mock state
_STATE: Dict[str, Any] = {
    "status": "idle",
    "progress": 0.0,
    "epoch": 0,
    "total_epochs": 0,
    "message": None,
    "started_at": None,
}


def _start(kind: str, config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    _STATE.update(
        status="running",
        progress=0.0,
        epoch=0,
        total_epochs=int((config or {}).get("epochs", 0) or 0),
        message=f"{kind} started (mock)",
        started_at=datetime.now().isoformat(),
    )
    logger.info("%s started (mock): %s", kind, config)
    return {"status": "started", "kind": kind, **_STATE}


@router.get("/status", response_model=TrainingStatus)
async def get_training_status():
    """Get current training status."""
    return TrainingStatus(**_STATE)


@router.post("/start")
async def start_training(config: Optional[Dict[str, Any]] = None):
    """Start a training run."""
    return _start("training", config)


@router.post("/stop")
async def stop_training():
    """Stop the current training run."""
    _STATE.update(status="idle", progress=0.0, message="training stopped (mock)")
    logger.info("Training stopped (mock)")
    return {"status": "stopped", **_STATE}


@router.post("/pretrain")
async def start_pretraining(config: Optional[Dict[str, Any]] = None):
    """Start a pretraining run (MAE recipe)."""
    return _start("pretraining", config)


@router.post("/lora")
async def start_lora_training(config: Optional[Dict[str, Any]] = None):
    """Start a LoRA fine-tuning run."""
    return _start("lora", config)
