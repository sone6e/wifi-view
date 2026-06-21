"""
FastAPI application for WiFi-DensePose API
"""

import asyncio
import logging
import logging.config
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.config.settings import get_settings
from src.config.domains import get_domain_config
from src.api.routers import pose, stream, health, auth, models, train, recording
from src.api.middleware.auth import AuthMiddleware
from src.api.middleware.rate_limit import RateLimitMiddleware
from src.api.dependencies import get_pose_service, get_stream_service, get_hardware_service
from src.api.websocket.connection_manager import connection_manager
from src.api.websocket.pose_stream import PoseStreamHandler
from src.bridge.udp_aggregator import UDPAggregator, get_aggregator

# Configure logging
settings = get_settings()
logging.config.dictConfig(settings.get_logging_config())
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting WiFi-DensePose API...")
    
    try:
        # Initialize services
        await initialize_services(app)
        
        # Start background tasks
        await start_background_tasks(app)
        
        logger.info("WiFi-DensePose API started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down WiFi-DensePose API...")
        await cleanup_services(app)
        logger.info("WiFi-DensePose API shutdown complete")


async def initialize_services(app: FastAPI):
    """Initialize application services."""
    try:
        # Initialize hardware service
        hardware_service = get_hardware_service()
        await hardware_service.initialize()
        
        # Initialize pose service
        pose_service = get_pose_service()
        await pose_service.initialize()
        
        # Initialize stream service
        stream_service = get_stream_service()
        await stream_service.initialize()
        
        # Initialize pose stream handler
        pose_stream_handler = PoseStreamHandler(
            connection_manager=connection_manager,
            pose_service=pose_service,
            stream_service=stream_service
        )
        
        # Store in app state for access in routes
        app.state.hardware_service = hardware_service
        app.state.pose_service = pose_service
        app.state.stream_service = stream_service
        app.state.pose_stream_handler = pose_stream_handler
        
        logger.info("Services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise


async def start_background_tasks(app: FastAPI):
    """Start background tasks."""
    try:
        # Start pose service
        pose_service = app.state.pose_service
        await pose_service.start()
        logger.info("Pose service started")
        
        # Start pose streaming if enabled
        if settings.enable_real_time_processing:
            pose_stream_handler = app.state.pose_stream_handler
            await pose_stream_handler.start_streaming()

        # Start ESP32 UDP aggregator (listens on port 5005 for CSI frames)
        aggregator = UDPAggregator(host='0.0.0.0', port=5005)
        try:
            await aggregator.start()
            app.state.udp_aggregator = aggregator
        except OSError as e:
            logger.warning(f"UDP aggregator failed to bind (port 5005 in use?): {e}")
        
        logger.info("Background tasks started")
        
    except Exception as e:
        logger.error(f"Failed to start background tasks: {e}")
        raise


async def cleanup_services(app: FastAPI):
    """Cleanup services on shutdown."""
    try:
        # Stop UDP aggregator
        if hasattr(app.state, 'udp_aggregator'):
            await app.state.udp_aggregator.stop()

        # Stop pose streaming
        if hasattr(app.state, 'pose_stream_handler'):
            await app.state.pose_stream_handler.shutdown()
        
        # Shutdown connection manager
        await connection_manager.shutdown()
        
        # Cleanup services
        if hasattr(app.state, 'stream_service'):
            await app.state.stream_service.shutdown()
        
        if hasattr(app.state, 'pose_service'):
            await app.state.pose_service.stop()
        
        if hasattr(app.state, 'hardware_service'):
            await app.state.hardware_service.shutdown()
        
        logger.info("Services cleaned up successfully")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="WiFi-based human pose estimation and activity recognition API",
    docs_url=settings.docs_url if not settings.is_production else None,
    redoc_url=settings.redoc_url if not settings.is_production else None,
    openapi_url=settings.openapi_url if not settings.is_production else None,
    lifespan=lifespan
)

# Add middleware
if settings.enable_rate_limiting:
    app.add_middleware(RateLimitMiddleware)

if settings.enable_authentication:
    app.add_middleware(AuthMiddleware)

# Add CORS middleware
cors_config = settings.get_cors_config()
app.add_middleware(
    CORSMiddleware,
    **cors_config
)

# Add trusted host middleware for production
if settings.is_production:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts
    )


# Exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "type": "http_error"
            }
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": 422,
                "message": "Validation error",
                "type": "validation_error",
                "details": exc.errors()
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "type": "internal_error"
            }
        }
    )


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    start_time = asyncio.get_event_loop().time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = asyncio.get_event_loop().time() - start_time
    
    # Log request
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    # Add processing time header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# Include routers
app.include_router(
    health.router,
    prefix="/health",
    tags=["Health"]
)

app.include_router(
    pose.router,
    prefix=f"{settings.api_prefix}/pose",
    tags=["Pose Estimation"]
)

app.include_router(
    stream.router,
    prefix=f"{settings.api_prefix}/stream",
    tags=["Streaming"]
)

app.include_router(
    auth.router,
    prefix=f"{settings.api_prefix}",
    tags=["Authentication"]
)

app.include_router(
    models.router,
    prefix=f"{settings.api_prefix}/models",
    tags=["Models"]
)

app.include_router(
    train.router,
    prefix=f"{settings.api_prefix}/train",
    tags=["Training"]
)

app.include_router(
    recording.router,
    prefix=f"{settings.api_prefix}/recording",
    tags=["Recording"]
)


@app.websocket("/ws/train/progress")
async def websocket_train_progress(websocket: WebSocket):
    """Stream training progress to the UI (mock: emits current training state)."""
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(train._STATE)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        logger.info("Training progress WebSocket disconnected")
    except Exception as e:
        logger.warning(f"Training progress WebSocket error: {e}")


@app.websocket("/ws/sensing")
async def websocket_sensing(websocket: WebSocket):
    """Stream WiFi sensing data to the UI — real ESP32 data or simulated fallback."""
    import math
    import time as _time
    from src.bridge.udp_aggregator import get_aggregator
    from src.bridge.frame_parser import RawCSIFrame, FeatureState

    await websocket.accept()
    try:
        while True:
            aggregator = get_aggregator()
            t = _time.time()

            # If we have real ESP32 data, use it
            if aggregator and aggregator.is_receiving:
                # Try feature state first (pre-processed on ESP32)
                if aggregator.feature_states:
                    fs = aggregator.feature_states[-1]
                    frame = {
                        "type": "sensing_update",
                        "timestamp": t,
                        "source": "esp32_live",
                        "nodes": [{
                            "node_id": fs.node_id,
                            "rssi_dbm": -45,
                            "position": [2, 0, 1.5],
                            "amplitude": [],
                            "subcarrier_count": 0,
                        }],
                        "features": {
                            "mean_rssi": -45,
                            "variance": fs.motion_score * 3,
                            "std": fs.motion_score * 1.5,
                            "motion_band_power": fs.motion_score,
                            "breathing_band_power": fs.respiration_conf * 0.2,
                            "dominant_freq_hz": fs.respiration_bpm / 60.0 if fs.respiration_bpm > 0 else 0.25,
                            "change_points": 0,
                            "spectral_power": fs.motion_score + fs.anomaly_score,
                            "range": fs.motion_score * 5,
                            "iqr": fs.motion_score * 2,
                            "skewness": 0.0,
                            "kurtosis": 1.0,
                        },
                        "classification": {
                            "motion_level": "present_moving" if fs.motion_score > 0.5 else ("present_still" if fs.presence_score > 0.5 else "absent"),
                            "presence": fs.presence_score > 0.3,
                            "confidence": fs.presence_score,
                        },
                        "vitals": {
                            "respiration_bpm": fs.respiration_bpm,
                            "respiration_confidence": fs.respiration_conf,
                            "heartbeat_bpm": fs.heartbeat_bpm,
                            "heartbeat_confidence": fs.heartbeat_conf,
                        },
                    }
                # Fall back to raw CSI frames
                elif aggregator.raw_frames:
                    raw = aggregator.raw_frames[-1]
                    amp_std = float(raw.amplitude.std()) if len(raw.amplitude) > 0 else 0
                    frame = {
                        "type": "sensing_update",
                        "timestamp": t,
                        "source": "esp32_raw",
                        "nodes": [{
                            "node_id": raw.node_id,
                            "rssi_dbm": raw.rssi,
                            "position": [2, 0, 1.5],
                            "amplitude": raw.amplitude[:64].tolist(),
                            "subcarrier_count": raw.num_subcarriers,
                        }],
                        "features": {
                            "mean_rssi": float(raw.rssi),
                            "variance": amp_std,
                            "std": amp_std,
                            "motion_band_power": min(amp_std / 100, 1.0),
                            "breathing_band_power": 0.05,
                            "dominant_freq_hz": 0.3,
                            "change_points": 0,
                            "spectral_power": amp_std / 50,
                            "range": amp_std * 3,
                            "iqr": amp_std * 1.5,
                            "skewness": 0.0,
                            "kurtosis": 1.0,
                        },
                        "classification": {
                            "motion_level": "present_still" if amp_std > 10 else "absent",
                            "presence": amp_std > 10,
                            "confidence": min(amp_std / 50, 0.95),
                        },
                    }
                else:
                    await asyncio.sleep(0.1)
                    continue
            else:
                # Simulated fallback when no ESP32 is connected
                base_rssi = -45
                variance = 1.5 + math.sin(t * 0.1)
                motion_band = 0.05 + abs(math.sin(t * 0.3)) * 0.15
                breath_band = 0.03 + abs(math.sin(t * 0.05)) * 0.08
                is_present = variance > 0.8
                frame = {
                    "type": "sensing_update",
                    "timestamp": t,
                    "source": "simulated",
                    "nodes": [{
                        "node_id": 1,
                        "rssi_dbm": base_rssi + math.sin(t * 0.5) * 3,
                        "position": [2, 0, 1.5],
                        "amplitude": [],
                        "subcarrier_count": 0,
                    }],
                    "features": {
                        "mean_rssi": base_rssi + math.sin(t * 0.5) * 3,
                        "variance": variance,
                        "std": math.sqrt(abs(variance)),
                        "motion_band_power": motion_band,
                        "breathing_band_power": breath_band,
                        "dominant_freq_hz": 0.3 + math.sin(t * 0.02) * 0.1,
                        "change_points": 0,
                        "spectral_power": motion_band + breath_band,
                        "range": variance * 3,
                        "iqr": variance * 1.5,
                        "skewness": 0.0,
                        "kurtosis": 1.0,
                    },
                    "classification": {
                        "motion_level": "present_still" if is_present else "absent",
                        "presence": is_present,
                        "confidence": 0.85 if is_present else 0.6,
                    },
                }

            await websocket.send_json(frame)
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        logger.info("Sensing WebSocket disconnected")
    except Exception as e:
        logger.warning(f"Sensing WebSocket error: {e}")


@app.websocket("/api/v1/stream/pose")
async def websocket_pose_stream(websocket: WebSocket):
    """Stream real-time pose inference results to the UI.

    When a model is loaded and ESP32 is sending CSI, this streams predicted
    keypoints/activity for PoseDetectionCanvas to render.
    """
    import time as _time
    from src.bridge.inference import get_inference_engine
    from src.bridge.udp_aggregator import get_aggregator

    await websocket.accept()
    try:
        while True:
            engine = get_inference_engine()
            aggregator = get_aggregator()

            if engine.is_loaded and aggregator and aggregator.raw_frames:
                # Real inference on latest CSI frame
                raw = aggregator.raw_frames[-1]
                result = engine.predict(raw.amplitude)
                if result:
                    result["timestamp"] = _time.time()
                    result["source"] = "esp32_inference"
                    await websocket.send_json(result)
                else:
                    await asyncio.sleep(0.1)
            elif engine.is_loaded:
                # Model loaded but no ESP32 data — send idle status
                await websocket.send_json({
                    "type": "pose_update",
                    "timestamp": _time.time(),
                    "source": "waiting_for_csi",
                    "persons": [],
                    "message": "Model loaded, waiting for ESP32 CSI data on UDP:5005",
                })
                await asyncio.sleep(1)
            else:
                # No model loaded
                await websocket.send_json({
                    "type": "pose_update",
                    "timestamp": _time.time(),
                    "source": "no_model",
                    "persons": [],
                    "message": "No model loaded. Record CSI → Train → Load model to see poses.",
                })
                await asyncio.sleep(2)

            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        logger.info("Pose stream WebSocket disconnected")
    except Exception as e:
        logger.warning(f"Pose stream WebSocket error: {e}")


@app.get(f"{settings.api_prefix}/bridge/status")
async def bridge_status():
    """Get ESP32 bridge status (UDP aggregator + inference engine)."""
    from src.bridge.udp_aggregator import get_aggregator
    from src.bridge.inference import get_inference_engine

    aggregator = get_aggregator()
    engine = get_inference_engine()

    return {
        "udp_aggregator": aggregator.get_stats() if aggregator else {"status": "not_started"},
        "inference_engine": {
            "model_loaded": engine.is_loaded,
            "model_id": engine.model_id,
            "model_type": engine.model_type,
            "device": engine.device,
        },
        "instructions": {
            "esp32_setup": "Configure ESP32 firmware: CSI_TARGET_IP=<your_pc_ip>, CSI_TARGET_PORT=5005",
            "workflow": "1. Record CSI → 2. Train model → 3. Load model → 4. See poses live",
        },
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
        "docs_url": settings.docs_url,
        "api_prefix": settings.api_prefix,
        "features": {
            "authentication": settings.enable_authentication,
            "rate_limiting": settings.enable_rate_limiting,
            "websockets": settings.enable_websockets,
            "real_time_processing": settings.enable_real_time_processing
        }
    }


# API information endpoint
@app.get(f"{settings.api_prefix}/info")
async def api_info():
    """Get detailed API information."""
    domain_config = get_domain_config()
    
    return {
        "api": {
            "name": settings.app_name,
            "version": settings.version,
            "environment": settings.environment,
            "prefix": settings.api_prefix
        },
        "configuration": {
            "zones": len(domain_config.zones),
            "routers": len(domain_config.routers),
            "pose_models": len(domain_config.pose_models)
        },
        "features": {
            "authentication": settings.enable_authentication,
            "rate_limiting": settings.enable_rate_limiting,
            "websockets": settings.enable_websockets,
            "real_time_processing": settings.enable_real_time_processing,
            "historical_data": settings.enable_historical_data
        },
        "limits": {
            "rate_limit_requests": settings.rate_limit_requests,
            "rate_limit_window": settings.rate_limit_window,
            "max_websocket_connections": domain_config.streaming.max_connections
        }
    }


# Status endpoint
@app.get(f"{settings.api_prefix}/status")
async def api_status(request: Request):
    """Get current API status."""
    try:
        # Get services from app state
        hardware_service = getattr(request.app.state, 'hardware_service', None)
        pose_service = getattr(request.app.state, 'pose_service', None)
        stream_service = getattr(request.app.state, 'stream_service', None)
        pose_stream_handler = getattr(request.app.state, 'pose_stream_handler', None)
        
        # Get service statuses
        status = {
            "api": {
                "status": "healthy",
                "uptime": "unknown",
                "version": settings.version
            },
            "services": {
                "hardware": await hardware_service.get_status() if hardware_service else {"status": "unavailable"},
                "pose": await pose_service.get_status() if pose_service else {"status": "unavailable"},
                "stream": await stream_service.get_status() if stream_service else {"status": "unavailable"}
            },
            "streaming": pose_stream_handler.get_stream_status() if pose_stream_handler else {"is_streaming": False},
            "connections": await connection_manager.get_connection_stats()
        }
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting API status: {e}")
        return {
            "api": {
                "status": "error",
                "error": str(e)
            }
        }


# Metrics endpoint (if enabled)
if settings.metrics_enabled:
    @app.get(f"{settings.api_prefix}/metrics")
    async def api_metrics(request: Request):
        """Get API metrics."""
        try:
            # Get services from app state
            pose_stream_handler = getattr(request.app.state, 'pose_stream_handler', None)
            
            metrics = {
                "connections": await connection_manager.get_metrics(),
                "streaming": await pose_stream_handler.get_performance_metrics() if pose_stream_handler else {}
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return {"error": str(e)}


# Development endpoints (only in development)
if settings.is_development and settings.enable_test_endpoints:
    @app.get(f"{settings.api_prefix}/dev/config")
    async def dev_config():
        """Get current configuration (development only).

        Returns a sanitized view -- secret keys and passwords are redacted.
        """
        _sensitive = {"secret", "password", "token", "key", "credential", "auth"}
        raw = settings.dict()
        sanitized = {
            k: "***REDACTED***" if any(s in k.lower() for s in _sensitive) else v
            for k, v in raw.items()
        }
        domain_config = get_domain_config()
        return {
            "settings": sanitized,
            "domain_config": domain_config.to_dict()
        }
    
    @app.post(f"{settings.api_prefix}/dev/reset")
    async def dev_reset(request: Request):
        """Reset services (development only)."""
        try:
            # Reset services
            hardware_service = getattr(request.app.state, 'hardware_service', None)
            pose_service = getattr(request.app.state, 'pose_service', None)
            
            if hardware_service:
                await hardware_service.reset()
            
            if pose_service:
                await pose_service.reset()
            
            return {"message": "Services reset successfully"}
            
        except Exception as e:
            logger.error(f"Error resetting services: {e}")
            return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        workers=settings.workers if not settings.reload else 1,
        log_level=settings.log_level.lower()
    )