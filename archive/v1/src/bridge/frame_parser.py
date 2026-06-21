"""Parse ADR-018 and ADR-081 binary frames from ESP32 nodes."""

import struct
import numpy as np
from dataclasses import dataclass
from typing import Optional

# Magic numbers
MAGIC_RAW_CSI = 0xC5110001      # ADR-018 raw CSI frame
MAGIC_VITALS = 0xC5110002       # ADR-039 vitals packet
MAGIC_FEATURE_VEC = 0xC5110003  # ADR-069 feature vector
MAGIC_FUSED_VITALS = 0xC5110004 # ADR-063 fused vitals
MAGIC_COMPRESSED = 0xC5110005   # ADR-039 compressed CSI
MAGIC_FEATURE_STATE = 0xC5110006  # ADR-081 feature state (60 bytes)
MAGIC_WASM_OUTPUT = 0xC5110007  # ADR-040 WASM output

# ADR-018 header: magic(4) + node_id(1) + channel(1) + bw(1) + antennas(1) +
#                 subcarriers(2) + rssi(1) + noise(1) + seq(2) + reserved(2) + timestamp(4) = 20 bytes
ADR018_HEADER_SIZE = 20

# ADR-081 feature state: 60 bytes packed
ADR081_SIZE = 60
ADR081_FORMAT = '<IB B H Q 9f H H I'  # little-endian packed struct


@dataclass
class RawCSIFrame:
    """Parsed ADR-018 raw CSI frame."""
    node_id: int
    channel: int
    bandwidth: int
    num_antennas: int
    num_subcarriers: int
    rssi: int
    noise_floor: int
    seq: int
    timestamp_us: int
    amplitude: np.ndarray  # shape: (num_antennas * num_subcarriers,)

    @property
    def snr(self) -> float:
        return self.rssi - self.noise_floor


@dataclass
class FeatureState:
    """Parsed ADR-081 feature state packet."""
    node_id: int
    mode: int
    seq: int
    timestamp_us: int
    motion_score: float
    presence_score: float
    respiration_bpm: float
    respiration_conf: float
    heartbeat_bpm: float
    heartbeat_conf: float
    anomaly_score: float
    env_shift_score: float
    node_coherence: float
    quality_flags: int


def parse_frame(data: bytes) -> Optional[RawCSIFrame | FeatureState]:
    """Parse a binary frame from ESP32. Returns None if unrecognized."""
    if len(data) < 4:
        return None

    magic = struct.unpack_from('<I', data, 0)[0]

    if magic == MAGIC_RAW_CSI:
        return _parse_raw_csi(data)
    elif magic == MAGIC_FEATURE_STATE:
        return _parse_feature_state(data)
    else:
        return None


def _parse_raw_csi(data: bytes) -> Optional[RawCSIFrame]:
    """Parse ADR-018 raw CSI frame."""
    if len(data) < ADR018_HEADER_SIZE:
        return None

    # Header: magic(4), node_id(1), channel(1), bw(1), antennas(1),
    #          subcarriers(2), rssi(1), noise(1), seq(2), reserved(2), timestamp(4)
    header = struct.unpack_from('<I B B B B H b b H H I', data, 0)
    _, node_id, channel, bw, antennas, subcarriers, rssi, noise, seq, _, ts = header

    # Payload: int16 amplitudes (antennas * subcarriers values)
    payload_size = antennas * subcarriers * 2
    if len(data) < ADR018_HEADER_SIZE + payload_size:
        # Partial frame — use what we have
        available = (len(data) - ADR018_HEADER_SIZE) // 2
        amplitude = np.frombuffer(data[ADR018_HEADER_SIZE:ADR018_HEADER_SIZE + available * 2], dtype=np.int16).astype(np.float32)
    else:
        amplitude = np.frombuffer(data[ADR018_HEADER_SIZE:ADR018_HEADER_SIZE + payload_size], dtype=np.int16).astype(np.float32)

    return RawCSIFrame(
        node_id=node_id,
        channel=channel,
        bandwidth=bw,
        num_antennas=antennas,
        num_subcarriers=subcarriers,
        rssi=rssi,
        noise_floor=noise,
        seq=seq,
        timestamp_us=ts,
        amplitude=amplitude,
    )


def _parse_feature_state(data: bytes) -> Optional[FeatureState]:
    """Parse ADR-081 feature state (60 bytes)."""
    if len(data) < ADR081_SIZE:
        return None

    # Unpack: magic(4) + node_id(1) + mode(1) + seq(2) + ts_us(8) +
    #         9 floats(36) + quality_flags(2) + reserved(2) + crc32(4) = 60
    values = struct.unpack_from('<I B B H Q 9f H H I', data, 0)
    (magic, node_id, mode, seq, ts_us,
     motion, presence, resp_bpm, resp_conf,
     hr_bpm, hr_conf, anomaly, env_shift, coherence,
     quality_flags, _reserved, _crc) = values

    return FeatureState(
        node_id=node_id,
        mode=mode,
        seq=seq,
        timestamp_us=ts_us,
        motion_score=motion,
        presence_score=presence,
        respiration_bpm=resp_bpm,
        respiration_conf=resp_conf,
        heartbeat_bpm=hr_bpm,
        heartbeat_conf=hr_conf,
        anomaly_score=anomaly,
        env_shift_score=env_shift,
        node_coherence=coherence,
        quality_flags=quality_flags,
    )
