"""CSI data recorder — saves frames to disk for model training.

Saves recordings as .npz files containing:
- amplitudes: (N, subcarriers) float32 array
- timestamps: (N,) float64 array of Unix timestamps
- metadata: dict with channel, bandwidth, node_id, etc.

For training with ground truth, use the companion camera recorder
to label poses simultaneously.
"""

import os
import time
import logging
import numpy as np
from pathlib import Path
from typing import Optional

from .frame_parser import RawCSIFrame

logger = logging.getLogger(__name__)

# Default storage path
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "recordings"


class CSIRecorder:
    """Records CSI frames to disk for model training."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.is_recording = False
        self.frames: list[RawCSIFrame] = []
        self.start_time: Optional[float] = None
        self.recording_id: Optional[str] = None
        self.label: Optional[str] = None

    def start(self, label: str = "unlabeled") -> str:
        """Start a new recording session. Returns recording ID."""
        self.is_recording = True
        self.frames = []
        self.start_time = time.time()
        self.recording_id = f"rec_{int(self.start_time)}"
        self.label = label
        logger.info(f"Recording started: {self.recording_id} (label={label})")
        return self.recording_id

    def add_frame(self, frame: RawCSIFrame):
        """Add a frame to the current recording."""
        if self.is_recording:
            self.frames.append(frame)

    def stop(self) -> dict:
        """Stop recording and save to disk. Returns recording metadata."""
        if not self.is_recording:
            return {"error": "Not recording"}

        self.is_recording = False
        duration = time.time() - (self.start_time or time.time())
        num_frames = len(self.frames)

        if num_frames == 0:
            logger.warning("Recording stopped with 0 frames")
            return {
                "id": self.recording_id,
                "frames": 0,
                "duration_s": duration,
                "file": None,
                "status": "empty",
            }

        # Build numpy arrays
        amplitudes = np.array([f.amplitude for f in self.frames], dtype=np.float32)
        timestamps = np.array([f.timestamp_us / 1e6 for f in self.frames], dtype=np.float64)

        # Metadata
        first = self.frames[0]
        metadata = {
            "recording_id": self.recording_id,
            "label": self.label,
            "node_id": int(first.node_id),
            "channel": int(first.channel),
            "bandwidth": int(first.bandwidth),
            "num_antennas": int(first.num_antennas),
            "num_subcarriers": int(first.num_subcarriers),
            "num_frames": num_frames,
            "duration_s": float(duration),
            "fps": float(num_frames / max(duration, 0.001)),
            "start_time": float(self.start_time),
        }

        # Save as .npz
        filename = f"{self.recording_id}_{self.label}.npz"
        filepath = self.data_dir / filename
        np.savez_compressed(
            filepath,
            amplitudes=amplitudes,
            timestamps=timestamps,
            metadata=metadata,
        )

        logger.info(f"Recording saved: {filepath} ({num_frames} frames, {duration:.1f}s)")

        return {
            "id": self.recording_id,
            "label": self.label,
            "frames": num_frames,
            "duration_s": round(duration, 1),
            "fps": round(num_frames / max(duration, 0.001), 1),
            "file": str(filepath),
            "status": "completed",
        }

    def list_recordings(self) -> list[dict]:
        """List all saved recordings."""
        recordings = []
        for f in sorted(self.data_dir.glob("rec_*.npz")):
            try:
                data = np.load(f, allow_pickle=True)
                meta = data['metadata'].item() if 'metadata' in data else {}
                recordings.append({
                    "id": meta.get("recording_id", f.stem),
                    "label": meta.get("label", "unknown"),
                    "frames": meta.get("num_frames", 0),
                    "duration_s": meta.get("duration_s", 0),
                    "fps": meta.get("fps", 0),
                    "file": str(f),
                    "size_kb": round(f.stat().st_size / 1024, 1),
                })
            except Exception as e:
                logger.warning(f"Error reading {f}: {e}")
        return recordings

    def delete_recording(self, recording_id: str) -> bool:
        """Delete a recording by ID."""
        for f in self.data_dir.glob(f"{recording_id}*.npz"):
            f.unlink()
            logger.info(f"Deleted recording: {f}")
            return True
        return False
