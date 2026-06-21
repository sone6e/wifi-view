"""UDP aggregator — listens for ESP32 CSI frames and feeds the pipeline.

The ESP32 firmware sends binary ADR-018/ADR-081 frames via UDP to
CONFIG_CSI_TARGET_IP:CONFIG_CSI_TARGET_PORT (default 192.168.1.100:5005).

This module:
1. Binds a UDP socket on port 5005 (configurable)
2. Parses incoming frames (raw CSI or feature state)
3. Stores frames in a shared buffer for recording/inference
4. Pushes sensing updates to the /ws/sensing WebSocket clients
"""

import asyncio
import logging
import time
from collections import deque
from typing import Callable, Optional

from .frame_parser import parse_frame, RawCSIFrame, FeatureState

logger = logging.getLogger(__name__)

# Global aggregator instance (singleton, started with the backend)
_aggregator: Optional['UDPAggregator'] = None


def get_aggregator() -> Optional['UDPAggregator']:
    return _aggregator


class UDPAggregator:
    """Async UDP server that receives ESP32 CSI frames."""

    def __init__(self, host: str = '0.0.0.0', port: int = 5005, buffer_size: int = 10000):
        self.host = host
        self.port = port
        self.buffer_size = buffer_size

        # Frame buffers
        self.raw_frames: deque = deque(maxlen=buffer_size)
        self.feature_states: deque = deque(maxlen=buffer_size)

        # Recording state
        self.is_recording = False
        self.recorded_frames: list = []
        self.recording_start_time: Optional[float] = None

        # Stats
        self.total_packets = 0
        self.parse_errors = 0
        self.last_frame_time: Optional[float] = None

        # Callbacks for real-time consumers
        self._on_frame_callbacks: list[Callable] = []
        self._on_feature_callbacks: list[Callable] = []

        self._transport = None
        self._protocol = None

    @property
    def is_receiving(self) -> bool:
        """True if we've received a frame in the last 3 seconds."""
        if self.last_frame_time is None:
            return False
        return (time.time() - self.last_frame_time) < 3.0

    def on_frame(self, callback: Callable):
        """Register a callback for raw CSI frames."""
        self._on_frame_callbacks.append(callback)

    def on_feature(self, callback: Callable):
        """Register a callback for feature state updates."""
        self._on_feature_callbacks.append(callback)

    def start_recording(self):
        """Start recording CSI frames for training."""
        self.is_recording = True
        self.recorded_frames = []
        self.recording_start_time = time.time()
        logger.info("CSI recording started")

    def stop_recording(self) -> list:
        """Stop recording and return captured frames."""
        self.is_recording = False
        frames = self.recorded_frames
        self.recorded_frames = []
        duration = time.time() - (self.recording_start_time or time.time())
        logger.info(f"CSI recording stopped: {len(frames)} frames in {duration:.1f}s")
        return frames

    def get_stats(self) -> dict:
        return {
            "total_packets": self.total_packets,
            "parse_errors": self.parse_errors,
            "is_receiving": self.is_receiving,
            "buffer_raw": len(self.raw_frames),
            "buffer_features": len(self.feature_states),
            "is_recording": self.is_recording,
            "recorded_frames": len(self.recorded_frames),
        }

    async def start(self):
        """Start the UDP listener."""
        global _aggregator
        _aggregator = self

        loop = asyncio.get_event_loop()
        self._transport, self._protocol = await loop.create_datagram_endpoint(
            lambda: _UDPProtocol(self),
            local_addr=(self.host, self.port)
        )
        logger.info(f"UDP aggregator listening on {self.host}:{self.port}")

    async def stop(self):
        """Stop the UDP listener."""
        global _aggregator
        if self._transport:
            self._transport.close()
        _aggregator = None
        logger.info("UDP aggregator stopped")

    def _handle_datagram(self, data: bytes, addr: tuple):
        """Process an incoming UDP datagram."""
        self.total_packets += 1
        self.last_frame_time = time.time()

        frame = parse_frame(data)
        if frame is None:
            self.parse_errors += 1
            return

        if isinstance(frame, RawCSIFrame):
            self.raw_frames.append(frame)
            if self.is_recording:
                self.recorded_frames.append(frame)
            for cb in self._on_frame_callbacks:
                try:
                    cb(frame)
                except Exception as e:
                    logger.warning(f"Frame callback error: {e}")

        elif isinstance(frame, FeatureState):
            self.feature_states.append(frame)
            for cb in self._on_feature_callbacks:
                try:
                    cb(frame)
                except Exception as e:
                    logger.warning(f"Feature callback error: {e}")


class _UDPProtocol(asyncio.DatagramProtocol):
    """asyncio DatagramProtocol adapter."""

    def __init__(self, aggregator: UDPAggregator):
        self.aggregator = aggregator

    def datagram_received(self, data: bytes, addr: tuple):
        self.aggregator._handle_datagram(data, addr)

    def error_received(self, exc: Exception):
        logger.warning(f"UDP error: {exc}")
