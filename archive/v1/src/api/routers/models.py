"""
Model management API endpoints.

In-memory mock implementation: lets the UI "Models" tab render and operate
(list / load / unload / active / LoRA / delete) without a real model registry
backing it. Replace the in-memory store with a real service when available.
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


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


# --- In-memory mock store ---

_MODELS: Dict[str, ModelInfo] = {
    "densepose_v1": ModelInfo(
        id="densepose_v1",
        name="DensePose V1",
        status="available",
        type="densepose",
        version="1.0.0",
    ),
}

_ACTIVE_MODEL: Dict[str, Optional[str]] = {"id": None}

_LORA_PROFILES: List[Dict[str, Any]] = []


# --- Endpoints ---

@router.get("", response_model=ModelListResponse)
@router.get("/", response_model=ModelListResponse)
async def list_models():
    """List all available models."""
    models = list(_MODELS.values())
    return ModelListResponse(models=models, count=len(models))


@router.get("/active")
async def get_active_model():
    """Get the currently active model (null if none loaded)."""
    active_id = _ACTIVE_MODEL["id"]
    if not active_id or active_id not in _MODELS:
        return None
    return _MODELS[active_id]


@router.get("/lora/profiles")
async def get_lora_profiles():
    """List available LoRA profiles."""
    return {"profiles": _LORA_PROFILES}


@router.post("/load")
async def load_model(request: LoadModelRequest):
    """Load a model by id."""
    if request.model_id not in _MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")
    _ACTIVE_MODEL["id"] = request.model_id
    logger.info("Model loaded (mock): %s", request.model_id)
    return {
        "status": "loaded",
        "model_id": request.model_id,
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/unload")
async def unload_model():
    """Unload the active model."""
    previous = _ACTIVE_MODEL["id"]
    _ACTIVE_MODEL["id"] = None
    logger.info("Model unloaded (mock): %s", previous)
    return {"status": "unloaded", "model_id": previous}


@router.post("/lora/activate")
async def activate_lora(request: LoraActivateRequest):
    """Activate a LoRA profile for a model."""
    if request.model_id not in _MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_id}' not found")
    logger.info("LoRA activated (mock): %s / %s", request.model_id, request.profile_name)
    return {
        "status": "activated",
        "model_id": request.model_id,
        "profile_name": request.profile_name,
    }


@router.get("/{model_id}")
async def get_model(model_id: str):
    """Get details for a single model."""
    if model_id not in _MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return _MODELS[model_id]


@router.delete("/{model_id}")
async def delete_model(model_id: str):
    """Delete a model from the registry."""
    if model_id not in _MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    del _MODELS[model_id]
    if _ACTIVE_MODEL["id"] == model_id:
        _ACTIVE_MODEL["id"] = None
    logger.info("Model deleted (mock): %s", model_id)
    return {"status": "deleted", "model_id": model_id}
