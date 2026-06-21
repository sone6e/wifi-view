"""
Model management API endpoints.

Scans `data/models/` for trained .rvf files and allows loading them for inference.
Also keeps the static DensePose V1 placeholder for compatibility.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()

MODEL_DIR = Path(__file__).parent.parent.parent.parent / "data" / "models"


# --- Response / request models ---

class ModelInfo(BaseModel):
    """Description of a single model."""

    id: str = Field(..., description="Unique model identifier")
    name: str = Field(..., description="Human-readable model name")
    status: str = Field(default="available", description="Model status")
    type: str = Field(default="densepose", description="Model type")
    version: str = Field(default="1.0.0", description="Model version")


class ModelListResponse(BaseModel):
    """Response model for listing models."""

    models: List[ModelInfo] = Field(default_factory=list)
    count: int = Field(default=0)


class LoadModelRequest(BaseModel):
    """Request body for loading a model."""

    model_id: str = Field(..., description="Model identifier to load")


class LoraActivateRequest(BaseModel):
    """Request body for activating a LoRA profile."""

    model_id: str = Field(..., description="Base model identifier")
    profile_name: str = Field(..., description="LoRA profile name")


_ACTIVE_MODEL: Dict[str, Optional[str]] = {"id": None}
_LORA_PROFILES: List[Dict[str, Any]] = []


def _scan_models() -> Dict[str, ModelInfo]:
    """Scan data/models/ for .rvf files and return model registry."""
    models: Dict[str, ModelInfo] = {}
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for f in sorted(MODEL_DIR.glob("*.rvf")):
        model_id = f.stem
        if "mae" in model_id:
            mtype = "mae_pretrained"
            name = f"MAE Encoder ({model_id})"
        elif "activity" in model_id:
            mtype = "activity_classifier"
            name = f"Activity Classifier ({model_id})"
        else:
            mtype = "densepose"
            name = model_id
        models[model_id] = ModelInfo(
            id=model_id,
            name=name,
            status="loaded" if _ACTIVE_MODEL["id"] == model_id else "available",
            type=mtype,
            version="1.0.0",
        )
    return models


# --- Endpoints ---

@router.get("", response_model=ModelListResponse)
@router.get("/", response_model=ModelListResponse)
async def list_models():
    """List all available models (scanned from data/models/)."""
    models = list(_scan_models().values())
    return ModelListResponse(models=models, count=len(models))


@router.get("/active")
async def get_active_model():
    """Get the currently active model (null if none loaded)."""
    active_id = _ACTIVE_MODEL["id"]
    if not active_id:
        return None
    models = _scan_models()
    if active_id not in models:
        return None
    return models[active_id]


@router.get("/lora/profiles")
async def get_lora_profiles():
    """List available LoRA profiles."""
    return {"profiles": _LORA_PROFILES}


@router.post("/load")
async def load_model(request: LoadModelRequest):
    """Load a model by id and start inference engine."""
    models = _scan_models()
    if request.model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")

    # Load into inference engine
    from src.bridge.inference import get_inference_engine
    engine = get_inference_engine()
    result = engine.load_model(request.model_id)
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message", "Load failed"))

    _ACTIVE_MODEL["id"] = request.model_id
    logger.info("Model loaded for inference: %s", request.model_id)
    return {
        "status": "loaded",
        "model_id": request.model_id,
        "model_type": result.get("type"),
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/unload")
async def unload_model():
    """Unload the active model and stop inference."""
    previous = _ACTIVE_MODEL["id"]
    _ACTIVE_MODEL["id"] = None

    from src.bridge.inference import get_inference_engine
    get_inference_engine().unload_model()

    logger.info("Model unloaded: %s", previous)
    return {"status": "unloaded", "model_id": previous}


@router.post("/lora/activate")
async def activate_lora(request: LoraActivateRequest):
    """Activate a LoRA profile for a model."""
    models = _scan_models()
    if request.model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")
    logger.info("LoRA activated: %s / %s", request.model_id, request.profile_name)
    return {
        "status": "activated",
        "model_id": request.model_id,
        "profile_name": request.profile_name,
    }


@router.get("/{model_id}")
async def get_model(model_id: str):
    """Get details for a single model."""
    models = _scan_models()
    if model_id not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return models[model_id]


@router.delete("/{model_id}")
async def delete_model(model_id: str):
    """Delete a model from the registry."""
    model_file = MODEL_DIR / f"{model_id}.rvf"
    if not model_file.exists():
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    model_file.unlink()
    if _ACTIVE_MODEL["id"] == model_id:
        _ACTIVE_MODEL["id"] = None
    logger.info("Model deleted: %s", model_id)
    return {"status": "deleted", "model_id": model_id}
