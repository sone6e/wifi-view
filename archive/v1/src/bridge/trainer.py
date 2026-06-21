"""CSI-to-Pose training pipeline.

Trains a neural network to predict human pose keypoints from WiFi CSI data.

Architecture: Simple CNN + FC head
  Input: (batch, 1, num_subcarriers) — CSI amplitude per frame
  Output: (batch, 17*2) — 17 COCO keypoints (x, y)

Training modes:
  1. Supervised: CSI frames paired with camera-derived keypoint labels
  2. Self-supervised pretraining (MAE): mask + reconstruct CSI subcarriers
  3. Activity classification: CSI → activity label (simpler, no camera needed)
"""

import logging
import time
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "recordings"
MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models"


def list_recordings() -> list[dict]:
    """List available training recordings."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    recordings = []
    for f in sorted(DATA_DIR.glob("rec_*.npz")):
        try:
            data = np.load(f, allow_pickle=True)
            meta = data['metadata'].item() if 'metadata' in data else {}
            recordings.append({
                "file": str(f),
                "id": meta.get("recording_id", f.stem),
                "label": meta.get("label", "unknown"),
                "frames": meta.get("num_frames", 0),
                "duration_s": meta.get("duration_s", 0),
            })
        except Exception as e:
            logger.warning(f"Error reading {f}: {e}")
    return recordings


def train_activity_classifier(
    recording_ids: Optional[list[str]] = None,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    device: str = "auto",
) -> dict:
    """Train an activity classifier from recorded CSI data.

    This is the simplest training mode — no camera labels needed.
    Each recording is treated as one activity class (based on its label).
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        return {"status": "error", "message": "PyTorch not installed. Run: pip install torch torchvision"}

    # Select device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on device: {device}")

    # Load recordings
    recordings = list_recordings()
    if recording_ids:
        recordings = [r for r in recordings if r["id"] in recording_ids]

    if len(recordings) < 2:
        return {
            "status": "error",
            "message": f"Need at least 2 recordings with different labels. Found {len(recordings)}. "
                       "Record CSI while doing different activities (e.g., 'standing', 'walking', 'sitting')."
        }

    # Build dataset
    all_features = []
    all_labels = []
    label_map = {}

    for rec in recordings:
        data = np.load(rec["file"], allow_pickle=True)
        amplitudes = data['amplitudes']  # (N, subcarriers)
        label = rec["label"]
        if label not in label_map:
            label_map[label] = len(label_map)
        label_idx = label_map[label]

        all_features.append(amplitudes)
        all_labels.extend([label_idx] * len(amplitudes))

    X = np.concatenate(all_features, axis=0)
    y = np.array(all_labels, dtype=np.int64)

    # Normalize
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-8
    X = (X - X_mean) / X_std

    # Train/val split (80/20)
    n = len(X)
    idx = np.random.permutation(n)
    split = int(n * 0.8)
    X_train, X_val = X[idx[:split]], X[idx[split:]]
    y_train, y_val = y[idx[:split]], y[idx[split:]]

    num_classes = len(label_map)
    input_dim = X.shape[1]

    logger.info(f"Training: {n} samples, {input_dim} features, {num_classes} classes: {label_map}")

    # Simple MLP model
    model = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, num_classes),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # DataLoader
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Training loop
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                correct += (pred.argmax(1) == yb).sum().item()
                total += len(yb)

        val_acc = correct / max(total, 1)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} — loss={train_loss/len(train_loader):.4f}, val_acc={val_acc:.3f}")

    duration = time.time() - start_time

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"activity_classifier_{int(time.time())}.rvf"
    torch.save({
        "model_state_dict": model.state_dict(),
        "label_map": label_map,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "normalization": {"mean": X_mean.tolist(), "std": X_std.tolist()},
        "architecture": "mlp_256_128",
        "best_val_acc": best_val_acc,
    }, model_path)

    logger.info(f"Model saved: {model_path} (val_acc={best_val_acc:.3f})")

    return {
        "status": "completed",
        "model_path": str(model_path),
        "model_id": model_path.stem,
        "accuracy": round(best_val_acc, 4),
        "epochs": epochs,
        "classes": label_map,
        "samples": n,
        "duration_s": round(duration, 1),
        "device": device,
    }


def train_mae_pretrain(
    recording_ids: Optional[list[str]] = None,
    epochs: int = 50,
    mask_ratio: float = 0.75,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    device: str = "auto",
) -> dict:
    """Self-supervised MAE pretraining on CSI data (no labels needed).

    Masks 75% of subcarrier values and trains a model to reconstruct them.
    The encoder can then be fine-tuned for pose estimation or activity recognition.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        return {"status": "error", "message": "PyTorch not installed. Run: pip install torch torchvision"}

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load all recordings
    recordings = list_recordings()
    if recording_ids:
        recordings = [r for r in recordings if r["id"] in recording_ids]

    if not recordings:
        return {"status": "error", "message": "No recordings found. Record CSI data first."}

    all_features = []
    for rec in recordings:
        data = np.load(rec["file"], allow_pickle=True)
        all_features.append(data['amplitudes'])

    X = np.concatenate(all_features, axis=0)
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-8
    X = (X - X_mean) / X_std

    input_dim = X.shape[1]
    n = len(X)
    logger.info(f"MAE pretraining: {n} frames, {input_dim} subcarriers, mask_ratio={mask_ratio}")

    # MAE model (simple encoder-decoder)
    encoder = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.GELU(),
        nn.Linear(256, 128),
        nn.GELU(),
    ).to(device)

    decoder = nn.Sequential(
        nn.Linear(128, 256),
        nn.GELU(),
        nn.Linear(256, input_dim),
    ).to(device)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=learning_rate)

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    start_time = time.time()
    best_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0
        for (xb,) in loader:
            xb = xb.to(device)
            # Random mask
            mask = torch.rand_like(xb) < mask_ratio
            masked_input = xb.clone()
            masked_input[mask] = 0

            # Forward
            encoded = encoder(masked_input)
            reconstructed = decoder(encoded)

            # Loss only on masked positions
            loss = ((reconstructed - xb) ** 2 * mask.float()).sum() / mask.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % 10 == 0:
            logger.info(f"MAE Epoch {epoch+1}/{epochs} — recon_loss={avg_loss:.6f}")

    duration = time.time() - start_time

    # Save pretrained encoder
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"mae_pretrained_{int(time.time())}.rvf"
    torch.save({
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "input_dim": input_dim,
        "latent_dim": 128,
        "normalization": {"mean": X_mean.tolist(), "std": X_std.tolist()},
        "architecture": "mae_256_128",
        "best_recon_loss": best_loss,
        "mask_ratio": mask_ratio,
    }, model_path)

    logger.info(f"MAE encoder saved: {model_path}")

    return {
        "status": "completed",
        "model_path": str(model_path),
        "model_id": model_path.stem,
        "reconstruction_loss": round(best_loss, 6),
        "epochs": epochs,
        "samples": n,
        "duration_s": round(duration, 1),
        "device": device,
    }
