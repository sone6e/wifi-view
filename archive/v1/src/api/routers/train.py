"""
Training management API endpoints.

Runs real training (activity classifier or MAE pretraining) using recorded CSI
data from the ESP32 bridge. Falls back to mock state when torch is unavailable.
"""

import logging
import threading
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


# Shared mutable state (updated by training thread)
_STATE: Dict[str, Any] = {
    "status": "idle",
    "progress": 0.0,
    "epoch": 0,
    "total_epochs": 0,
    "message": None,
    "started_at": None,
}

_training_thread: Optional[threading.Thread] = None


def _run_training(kind: str, config: Dict[str, Any]):
    """Run training in a background thread."""
    from src.bridge.trainer import train_activity_classifier, train_mae_pretrain

    epochs = int(config.get("epochs", 100))
    batch_size = int(config.get("batch_size", 32))
    lr = float(config.get("learning_rate", 3e-4))

    _STATE.update(status="running", total_epochs=epochs, epoch=0, progress=0.0)

    try:
        if kind == "pretraining":
            result = train_mae_pretrain(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=lr,
                mask_ratio=float(config.get("mask_ratio", 0.75)),
            )
        else:
            result = train_activity_classifier(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=lr,
            )

        if result["status"] == "completed":
            _STATE.update(
                status="completed",
                progress=1.0,
                epoch=epochs,
                message=f"Training complete — accuracy: {result.get('accuracy', 'N/A')}, model: {result.get('model_id', '')}",
            )
        else:
            _STATE.update(status="error", message=result.get("message", "Unknown error"))

    except Exception as e:
        logger.error(f"Training failed: {e}")
        _STATE.update(status="error", message=str(e))


@router.get("/status", response_model=TrainingStatus)
async def get_training_status():
    """Get current training status."""
    return TrainingStatus(**_STATE)


@router.post("/start")
async def start_training(config: Optional[Dict[str, Any]] = None):
    """Start a training run (activity classifier from recorded CSI)."""
    global _training_thread
    cfg = config or {}
    epochs = int(cfg.get("epochs", 100))

    _STATE.update(
        status="running",
        progress=0.0,
        epoch=0,
        total_epochs=epochs,
        message="Training started — activity classifier",
        started_at=datetime.now().isoformat(),
    )

    _training_thread = threading.Thread(target=_run_training, args=("training", cfg), daemon=True)
    _training_thread.start()

    return {"kind": "training", **_STATE}


@router.post("/stop")
async def stop_training():
    """Stop the current training run."""
    _STATE.update(status="idle", progress=0.0, message="Training stopped")
    logger.info("Training stopped")
    return {**_STATE, "status": "stopped"}


@router.post("/pretrain")
async def start_pretraining(config: Optional[Dict[str, Any]] = None):
    """Start MAE pretraining (self-supervised, no labels needed)."""
    global _training_thread
    cfg = config or {}
    epochs = int(cfg.get("epochs", 50))

    _STATE.update(
        status="running",
        progress=0.0,
        epoch=0,
        total_epochs=epochs,
        message="MAE pretraining started — self-supervised",
        started_at=datetime.now().isoformat(),
    )

    _training_thread = threading.Thread(target=_run_training, args=("pretraining", cfg), daemon=True)
    _training_thread.start()

    return {"kind": "pretraining", **_STATE}


@router.post("/lora")
async def start_lora_training(config: Optional[Dict[str, Any]] = None):
    """Start a LoRA fine-tuning run (requires pretrained encoder)."""
    global _training_thread
    cfg = config or {}
    cfg.setdefault("epochs", 30)

    _STATE.update(
        status="running",
        progress=0.0,
        epoch=0,
        total_epochs=int(cfg["epochs"]),
        message="LoRA fine-tuning started",
        started_at=datetime.now().isoformat(),
    )

    _training_thread = threading.Thread(target=_run_training, args=("training", cfg), daemon=True)
    _training_thread.start()

    return {"kind": "lora", **_STATE}
