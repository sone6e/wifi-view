"""Real-time inference — loads a trained model and predicts from live CSI.

When a model is loaded and the ESP32 is streaming CSI, this module:
1. Receives each CSI frame from the UDP aggregator
2. Passes it through the loaded model
3. Outputs predictions (activity class or keypoints)
4. Pushes results to the WebSocket for the frontend to render
"""

import logging
import time
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models"

# COCO 17-keypoint skeleton definition
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


class PoseInferenceEngine:
    """Loads a trained .rvf model and runs inference on CSI frames."""

    def __init__(self):
        self.model = None
        self.model_id: Optional[str] = None
        self.model_type: Optional[str] = None
        self.label_map: Optional[dict] = None
        self.normalization: Optional[dict] = None
        self.device = "cpu"
        self._torch = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load_model(self, model_id: str) -> dict:
        """Load a .rvf model file for inference."""
        try:
            import torch
            self._torch = torch
        except ImportError:
            return {"status": "error", "message": "PyTorch not installed"}

        model_path = MODEL_DIR / f"{model_id}.rvf"
        if not model_path.exists():
            return {"status": "error", "message": f"Model file not found: {model_path}"}

        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        except Exception as e:
            return {"status": "error", "message": f"Failed to load model: {e}"}

        self.model_id = model_id
        self.normalization = checkpoint.get("normalization")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Determine model type and reconstruct architecture
        if "model_state_dict" in checkpoint:
            # Activity classifier
            self.model_type = "activity_classifier"
            self.label_map = checkpoint.get("label_map", {})
            input_dim = checkpoint.get("input_dim", 256)
            num_classes = checkpoint.get("num_classes", 2)

            import torch.nn as nn
            model = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_classes),
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            self.model = model.to(self.device)

        elif "encoder_state_dict" in checkpoint:
            # MAE encoder (for feature extraction / visualization)
            self.model_type = "mae_encoder"
            self.label_map = None
            input_dim = checkpoint.get("input_dim", 256)

            import torch.nn as nn
            encoder = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.GELU(),
                nn.Linear(256, 128),
                nn.GELU(),
            )
            encoder.load_state_dict(checkpoint["encoder_state_dict"])
            encoder.eval()
            self.model = encoder.to(self.device)
        else:
            return {"status": "error", "message": "Unknown model format in .rvf file"}

        logger.info(f"Model loaded: {model_id} (type={self.model_type}, device={self.device})")
        return {"status": "loaded", "model_id": model_id, "type": self.model_type}

    def unload_model(self):
        """Unload the current model."""
        self.model = None
        self.model_id = None
        self.model_type = None
        self.label_map = None

    def predict(self, amplitude: np.ndarray) -> Optional[dict]:
        """Run inference on a single CSI frame.

        Args:
            amplitude: (num_subcarriers,) float32 array

        Returns:
            Prediction dict with type-specific fields, or None if no model loaded.
        """
        if not self.is_loaded or self._torch is None:
            return None

        torch = self._torch

        # Normalize
        if self.normalization:
            mean = np.array(self.normalization["mean"]).flatten()
            std = np.array(self.normalization["std"]).flatten()
            # Handle dimension mismatch (model trained on different subcarrier count)
            if len(mean) != len(amplitude):
                # Pad or truncate
                if len(amplitude) > len(mean):
                    amplitude = amplitude[:len(mean)]
                else:
                    padded = np.zeros(len(mean), dtype=np.float32)
                    padded[:len(amplitude)] = amplitude
                    amplitude = padded
            amplitude = (amplitude - mean) / std

        # Convert to tensor
        x = torch.tensor(amplitude, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(x)

        if self.model_type == "activity_classifier":
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()
            pred_class = int(probs.argmax())
            # Reverse label map
            inv_map = {v: k for k, v in (self.label_map or {}).items()}
            activity = inv_map.get(pred_class, f"class_{pred_class}")
            confidence = float(probs[pred_class])

            # Generate synthetic keypoints based on activity classification
            keypoints = self._activity_to_keypoints(activity, confidence)

            return {
                "type": "pose_update",
                "activity": activity,
                "confidence": confidence,
                "probabilities": {inv_map.get(i, f"class_{i}"): float(p) for i, p in enumerate(probs)},
                "persons": [{
                    "id": 0,
                    "keypoints": keypoints,
                    "confidence": confidence,
                    "activity": activity,
                }],
            }

        elif self.model_type == "mae_encoder":
            # MAE encoder outputs latent vector — use for visualization
            latent = output[0].cpu().numpy()
            return {
                "type": "latent_update",
                "latent_vector": latent.tolist(),
                "dimensions": len(latent),
            }

        return None

    def _activity_to_keypoints(self, activity: str, confidence: float) -> list:
        """Generate approximate COCO keypoints based on detected activity.

        These are placeholder poses — real pose estimation from CSI requires
        a specifically trained regression model (future enhancement).
        """
        # Base standing pose (normalized 0-1 coordinates)
        base_keypoints = {
            "standing": [
                [0.5, 0.1], [0.48, 0.08], [0.52, 0.08], [0.45, 0.09], [0.55, 0.09],
                [0.4, 0.25], [0.6, 0.25], [0.35, 0.4], [0.65, 0.4],
                [0.33, 0.55], [0.67, 0.55], [0.43, 0.55], [0.57, 0.55],
                [0.43, 0.75], [0.57, 0.75], [0.43, 0.95], [0.57, 0.95],
            ],
            "sitting": [
                [0.5, 0.2], [0.48, 0.18], [0.52, 0.18], [0.45, 0.19], [0.55, 0.19],
                [0.4, 0.35], [0.6, 0.35], [0.35, 0.5], [0.65, 0.5],
                [0.33, 0.6], [0.67, 0.6], [0.43, 0.55], [0.57, 0.55],
                [0.4, 0.7], [0.6, 0.7], [0.35, 0.8], [0.65, 0.8],
            ],
            "walking": [
                [0.5, 0.1], [0.48, 0.08], [0.52, 0.08], [0.45, 0.09], [0.55, 0.09],
                [0.4, 0.25], [0.6, 0.25], [0.3, 0.35], [0.7, 0.45],
                [0.25, 0.5], [0.72, 0.55], [0.43, 0.55], [0.57, 0.55],
                [0.4, 0.75], [0.6, 0.7], [0.38, 0.95], [0.62, 0.9],
            ],
        }

        # Default to standing if activity not in presets
        kp_coords = base_keypoints.get(activity, base_keypoints["standing"])

        # Add slight noise based on time for natural movement
        t = time.time()
        noise = np.sin(t * 2) * 0.01

        keypoints = []
        for i, (x, y) in enumerate(kp_coords):
            keypoints.append({
                "name": COCO_KEYPOINTS[i],
                "x": x + noise * (i % 3 - 1),
                "y": y + noise * ((i + 1) % 3 - 1),
                "confidence": confidence * (0.8 + 0.2 * np.random.random()),
            })

        return keypoints


# Singleton inference engine
_engine = PoseInferenceEngine()


def get_inference_engine() -> PoseInferenceEngine:
    return _engine
